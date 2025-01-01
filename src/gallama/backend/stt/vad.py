from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict
import torch
from silero_vad import VADIterator, load_silero_vad
from .audio_buffer import AudioBufferWithTiming
from ...data_classes.realtime_client_proto import TurnDetectionConfig
from gallama.logger import logger


class VADProcessor:
    def __init__(self, config):
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
        self.min_silence_ms = 500  # Minimum 500ms silence to end speech
        self.window_size_samples = 512
        self.window_stride_samples = 512  # Can be adjusted if needed

        # Event state tracking
        self.is_speech = False
        self.speech_start_sent = False
        self.speech_end_sent = False

    def get_windows_from_buffer(self, audio_buffer: AudioBufferWithTiming, current_offset: int):
        """Extract windows from audio buffer with proper timing."""
        audio_chunk = audio_buffer.get_unprocessed_audio()
        if len(audio_chunk) == 0:
            return []

        windows = []
        chunk_start_time_ms = audio_buffer.get_time_ms(current_offset)

        for i in range(0, len(audio_chunk) - self.window_size_samples + 1, self.window_stride_samples):
            window = audio_chunk[i:i + self.window_size_samples]
            if len(window) == self.window_size_samples:
                window_time_ms = chunk_start_time_ms + (i / self.sampling_rate * 1000)
                windows.append({
                    'data': torch.from_numpy(window).float(),
                    'time_ms': window_time_ms,
                    'index': i // self.window_stride_samples
                })

        return windows

    def process_chunk(self, audio_buffer: AudioBufferWithTiming, current_offset: int) -> Dict:
        windows = self.get_windows_from_buffer(audio_buffer, current_offset)
        if not windows:
            return {'speech_detected': False, 'speech_ended': False}

        speech_detected = False
        prob_list = []

        for window in windows:
            prob = self.model(window['data'], self.sampling_rate).item()
            logger.info(
                f"Window {window['index']} at {window['time_ms']:.3f}ms: prob={prob:.3f}, threshold={self.config.threshold}")
            prob_list.append(prob)

            if prob >= self.config.threshold:
                speech_detected = True
                self.last_speech_ms = window['time_ms']
                if not self.is_speech:
                    self.is_speech = True
                    self.speech_start_ms = window['time_ms']
                    self.continuous_silence_ms = 0

        avg_prob = sum(prob_list) / len(prob_list) if prob_list else 0.0
        logger.info(
            f"Chunk summary: windows={len(windows)}, avg_prob={avg_prob:.3f}, speech_detected={speech_detected}")

        if self.is_speech and not self.speech_start_sent:
            self.speech_start_sent = True
            logger.info(f"Speech start detected at {self.speech_start_ms}ms")
            return {
                'speech_detected': True,
                'speech_ended': False,
                'start_time': self.speech_start_ms,
                'end_time': None,
                'confidence': avg_prob
            }

        if self.is_speech and not speech_detected:
            self.continuous_silence_ms += (len(windows) * self.window_stride_samples / self.sampling_rate * 1000)

            if self.continuous_silence_ms >= self.min_silence_ms and not self.speech_end_sent:
                self.speech_end_sent = True
                logger.info(f"Speech end detected at {self.last_speech_ms}ms")
                return {
                    'speech_detected': True,
                    'speech_ended': True,
                    'start_time': self.speech_start_ms,
                    'end_time': self.last_speech_ms,
                    'duration_ms': self.last_speech_ms - self.speech_start_ms,
                    'confidence': avg_prob
                }
        else:
            self.continuous_silence_ms = 0

        return {
            'speech_detected': speech_detected,
            'speech_ended': False,
            'start_time': self.speech_start_ms if self.is_speech else None,
            'end_time': None,
            'confidence': avg_prob
        }

    def _update_speech_state(self, speech_detected: bool, speech_prob: float,
                             first_speech_time_ms: Optional[float],
                             last_speech_time_ms: Optional[float]) -> Dict:
        """Update internal speech state based on VAD results."""

        if not self.is_speech and speech_detected and first_speech_time_ms is not None:
            # Speech start detected
            self.is_speech = True
            self.speech_start_ms = first_speech_time_ms
            self.continuous_silence_ms = 0

            logger.info(f"Speech start detected at {self.speech_start_ms}ms")
            return {
                'speech_detected': True,
                'speech_ended': False,
                'start_time': self.speech_start_ms,
                'end_time': None,
                'confidence': speech_prob
            }

        elif self.is_speech:
            if speech_detected and last_speech_time_ms is not None:
                # Ongoing speech - update last speech time and reset silence counter
                self.last_speech_ms = last_speech_time_ms
                self.continuous_silence_ms = 0
            else:
                # Add time since last window
                self.continuous_silence_ms += self.window_stride_ms

                # Check if we've reached minimum silence duration
                if self.continuous_silence_ms >= self.min_silence_ms:
                    logger.info(f"Speech end detected after {self.continuous_silence_ms}ms silence")
                    logger.info(f"Speech segment: {self.speech_start_ms}ms -> {self.last_speech_ms}ms")

                    self.is_speech = False
                    return {
                        'speech_detected': True,
                        'speech_ended': True,
                        'start_time': self.speech_start_ms,
                        'end_time': self.last_speech_ms,
                        'duration_ms': self.last_speech_ms - self.speech_start_ms,
                        'confidence': speech_prob
                    }

        return {
            'speech_detected': self.is_speech,
            'speech_ended': False,
            'start_time': self.speech_start_ms if self.is_speech else None,
            'end_time': None,
            'confidence': speech_prob if self.is_speech else 0.0
        }

    def reset(self):
        """Reset VAD state."""
        if self.vad_iterator:
            self.vad_iterator.reset_states()

        # Reset timing state
        self.speech_start_ms = None
        self.last_speech_ms = None
        self.continuous_silence_ms = 0

        # Reset detection state
        self.is_speech = False
        self.speech_start_sent = False
        self.speech_end_sent = False