from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict, Tuple
import torch
from silero_vad import VADIterator, load_silero_vad
from .audio_buffer import AudioBufferWithTiming
from ...data_classes.realtime_client_proto import TurnDetectionConfig
from gallama.logger import logger
import copy



class VADProcessor:
    def __init__(self, config: TurnDetectionConfig):
        self.config = config
        self.sampling_rate = 16000

        # Initialize Silero VAD
        try:
            self.model = load_silero_vad()
            self.model.eval()
            self.vad_iterator = VADIterator(
                self.model,
                sampling_rate=self.sampling_rate,
                threshold=self.config.threshold
            )
        except Exception as e:
            logger.error(f"Error initializing Silero VAD: {str(e)}")
            raise

        # Speech detection state
        self.speech_start_ms = None
        self.last_speech_ms = None
        self.continuous_silence_ms = 0
        self.continuous_speaking_ms = 0

        logger.info(f"VADProcessor initialized with threshold={self.config.threshold}")
        self.min_silence_ms = self.config.silence_duration_ms  # Use config value for silence duration
        logger.info(f"Min silence duration: {self.min_silence_ms}ms")
        self.min_speak_ms = self.config.prefix_padding_ms
        self.window_size_samples = 512  # Process audio in 512-sample chunks
        self.window_stride_samples = 512  # No overlap between chunks

        # Event state tracking
        self.is_speaking = False
        self.speech_start_sent = False
        self.speech_end_sent = False
        self.prob_speech_start = None
        self.prob_speech_end = None
        self.potential_speech_start_flag = None
        self.potential_speech_end_flag = None

    @property
    def audio_time_offset(self) -> float:
        """
        Get the current audio time offset in seconds.
        This represents the start time of the current audio buffer.
        """
        return self.audio_buffer.start_offset / self.SAMPLING_RATE

    def get_windows_from_buffer(self, audio_buffer: AudioBufferWithTiming, current_offset: int):
        """Extract 512-sample windows from audio buffer with proper timing."""
        audio_chunk = audio_buffer.get_unprocessed_audio()
        if len(audio_chunk) == 0:
            return []

        windows = []
        chunk_start_time_ms = audio_buffer.get_time_ms(current_offset)

        # Process only full 512-sample windows
        for i in range(0, len(audio_chunk) - self.window_size_samples + 1, self.window_stride_samples):
            window = audio_chunk[i:i + self.window_size_samples]
            if len(window) == self.window_size_samples:  # Ensure full 512-sample windows
                window_time_ms = chunk_start_time_ms + (i / self.sampling_rate * 1000)
                windows.append({
                    'data': torch.from_numpy(window).float(),
                    'time_ms': window_time_ms,
                    'index': i // self.window_stride_samples
                })

        return windows

    def process_chunk(self, audio_buffer: AudioBufferWithTiming, current_offset: int, is_final: bool = False) -> Tuple[
        Dict, Dict]:
        windows = self.get_windows_from_buffer(audio_buffer, current_offset)
        if not windows:
            return None, None

        speech_start_object = None
        speech_end_object = None

        for window in windows:
            prob = self.model(window['data'], self.sampling_rate).item()
            logger.info(
                f"Window {window['index']} at {window['time_ms']:.3f}ms: prob={prob:.3f}, threshold={self.config.threshold}, continuous_silence_ms={self.continuous_silence_ms:.3f}"
            )

            if prob >= self.config.threshold:
                self.last_speech_ms = window['time_ms']
                if not self.is_speaking:
                    if not self.potential_speech_start_flag:
                        self.potential_speech_start_flag = True
                        self.speech_start_ms = window['time_ms']  # Track potential start time

                    # Accumulate speaking duration
                    self.continuous_speaking_ms += (self.window_size_samples / self.sampling_rate * 1000)

                    # Confirm speech start if min_speak_ms is achieved
                    if self.continuous_speaking_ms >= self.min_speak_ms:
                        self.is_speaking = True
                        self.speech_start_sent = True
                        self.prob_speech_start = prob
                        # Use absolute timing from the audio buffer
                        speech_start_object = {
                            'speech_detected': True,
                            'speech_ended': False,
                            'start_time': self.speech_start_ms,
                            'end_time': None,
                            'confidence': prob
                        }
                else:
                    # Reset potential speech end flag if speech continues
                    if self.potential_speech_end_flag:
                        logger.info(f"Reset potential_speech_end_flag because speech continues.")
                        self.potential_speech_end_flag = False
                        self.continuous_silence_ms = 0

            else:
                # Accumulate silence duration
                if self.is_speaking:
                    self.continuous_silence_ms += (self.window_size_samples / self.sampling_rate * 1000)

                # Check for potential speech end
                if not self.potential_speech_end_flag and self.is_speaking:
                    self.potential_speech_end_flag = True

                # Confirm speech end if silence duration exceeds min_silence_ms
                if self.continuous_silence_ms > self.min_silence_ms:
                    self.prob_speech_end = prob
                    # Use absolute timing from the audio buffer
                    speech_end_object = {
                        'speech_detected': True,
                        'speech_ended': True,
                        'start_time': self.speech_start_ms,
                        'end_time': self.last_speech_ms,
                        'duration_ms': self.last_speech_ms - self.speech_start_ms,
                        'confidence': 1 - prob
                    }
                    self.reset()
                    break  # Once speech end is confirmed, disregard the rest of the audio

        return speech_start_object, speech_end_object

    def reset(self):
        """Reset VAD state."""
        if self.vad_iterator:
            self.vad_iterator.reset_states()

        # Reset timing state
        self.speech_start_ms = None
        self.last_speech_ms = None
        self.continuous_silence_ms = 0
        self.continuous_speaking_ms = 0

        # Reset detection state
        self.is_speaking = False
        self.speech_start_sent = False
        self.speech_end_sent = False
        self.is_speaking = False
        self.speech_start_sent = False
        self.speech_end_sent = False
        self.prob_speech_start = None
        self.prob_speech_end = None
        self.potential_speech_start_flag = None
        self.potential_speech_end_flag = None