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
        audio_chunk = audio_buffer.get_unprocessed_audio()
        if len(audio_chunk) == 0:
            return []

        windows = []
        # Remove local time calculation, rely on AudioBufferWithTiming
        for i in range(0, len(audio_chunk) - self.window_size_samples + 1, self.window_stride_samples):
            window = audio_chunk[i:i + self.window_size_samples]
            if len(window) == self.window_size_samples:
                windows.append({
                    'data': torch.from_numpy(window).float(),
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
            # Get absolute time accounting for buffer offset
            absolute_time_ms = audio_buffer.get_time_ms(current_offset + window['index'] * self.window_stride_samples)

            prob = self.model(window['data'], self.sampling_rate).item()

            if prob >= self.config.threshold:
                self.last_speech_ms = absolute_time_ms
                if not self.is_speaking:
                    if not self.potential_speech_start_flag:
                        self.potential_speech_start_flag = True
                        self.speech_start_ms = absolute_time_ms  # Use absolute time

                    self.continuous_speaking_ms += (self.window_size_samples / self.sampling_rate * 1000)

                    if self.continuous_speaking_ms >= self.min_speak_ms:
                        self.is_speaking = True
                        self.speech_start_sent = True
                        self.prob_speech_start = prob
                        speech_start_object = {
                            'speech_detected': True,
                            'speech_ended': False,
                            'start_time': self.speech_start_ms,  # Already absolute time
                            'end_time': None,
                            'confidence': prob
                        }
                else:
                    if self.potential_speech_end_flag:
                        self.potential_speech_end_flag = False
                        self.continuous_silence_ms = 0

            else:
                if self.is_speaking:
                    self.continuous_silence_ms += (self.window_size_samples / self.sampling_rate * 1000)

                if not self.potential_speech_end_flag and self.is_speaking:
                    self.potential_speech_end_flag = True

                if self.continuous_silence_ms > self.min_silence_ms:
                    self.prob_speech_end = prob
                    speech_end_object = {
                        'speech_detected': True,
                        'speech_ended': True,
                        'start_time': self.speech_start_ms,  # Already absolute time
                        'end_time': self.last_speech_ms,  # Already absolute time
                        'duration_ms': self.last_speech_ms - self.speech_start_ms,
                        'confidence': 1 - prob
                    }
                    self.reset()
                    break

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