from silero_vad import load_silero_vad, VADIterator
from ..logger.logger import logger
from ..data_classes.realtime_data_classes import TurnDetectionConfig
import numpy as np
from typing import Tuple, Optional, Dict
import torch
from dataclasses import dataclass
from collections import deque


@dataclass
class VADState:
    is_speech: bool = False
    speech_start_time: float = 0.0
    accumulated_silence: float = 0.0
    last_window_time: float = 0.0
    potential_speech_start: float = 0.0  # Track when potential speech started
    speech_duration: float = 0.0  # Track current speech duration


class VADProcessor:
    def __init__(self, turn_detection_config: TurnDetectionConfig):
        self.model = load_silero_vad()
        self.model.eval()

        # Configuration
        self.threshold = turn_detection_config.threshold
        self.silence_duration_ms = turn_detection_config.silence_duration_ms
        self.prefix_padding_ms = turn_detection_config.prefix_padding_ms
        self.min_speech_duration_ms = 250  # Hardcoded minimum speech duration
        self.create_response = turn_detection_config.create_response

        # Fixed parameters
        self.sampling_rate = 16000
        self.window_size_samples = 512
        self.speech_pad_ms = 30

        # State management
        self.state = VADState()
        # Increase buffer size to accommodate prefix padding
        max_buffer_samples = int(self.sampling_rate * (self.prefix_padding_ms / 1000 + 2))
        self.audio_buffer = deque(maxlen=max_buffer_samples)
        self.current_chunk = np.array([], dtype=np.float32)

        # Initialize VAD iterator
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_silence_duration_ms=self.silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms
        )
        self.total_audio_processed = 0.0

    def reset(self):
        """Reset all state"""
        self.vad_iterator.reset_states()
        self.current_chunk = np.array([], dtype=np.float32)
        self.state = VADState()
        self.audio_buffer.clear()
        self.total_audio_processed = 0.0

    def _convert_audio(self, audio_chunk: bytes) -> np.ndarray:
        """Convert audio bytes to normalized float array"""
        try:
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            return audio_np.astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            return np.array([], dtype=np.float32)

    def _get_prefix_audio(self) -> np.ndarray:
        """Get prefix padding audio from buffer"""
        prefix_samples = int(self.sampling_rate * (self.prefix_padding_ms / 1000))
        buffer_array = np.array(list(self.audio_buffer), dtype=np.float32)

        if len(buffer_array) >= prefix_samples:
            return buffer_array[-prefix_samples:]
        return buffer_array

    def _update_speech_state(self, is_speech: bool, window_time: float) -> Optional[Dict[str, float]]:
        """Update speech state and return speech events"""
        event = None

        # Update total audio processed time
        self.total_audio_processed += self.window_size_samples / self.sampling_rate

        if is_speech:
            if not self.state.is_speech:
                # Start tracking potential speech
                if self.state.potential_speech_start == 0.0:
                    self.state.potential_speech_start = self.total_audio_processed

                # Calculate current speech duration
                current_duration = (
                                               self.total_audio_processed - self.state.potential_speech_start) * 1000  # Convert to ms

                # Check if speech duration meets minimum requirement
                if current_duration >= self.min_speech_duration_ms:
                    # Actual speech start
                    self.state.is_speech = True
                    self.state.speech_start_time = self.state.potential_speech_start
                    event = {'start': self.state.speech_start_time}
                    logger.debug(f"Speech started after reaching minimum duration: {current_duration:.1f}ms")

            # Reset silence counter when speech is detected
            self.state.accumulated_silence = 0

        else:
            if not self.state.is_speech:
                # Reset potential speech tracking if we haven't met minimum duration
                self.state.potential_speech_start = 0.0

            if self.state.is_speech:
                # Add fixed window size instead of using time_delta
                self.state.accumulated_silence += self.window_size_samples / self.sampling_rate

                if self.state.accumulated_silence >= (self.silence_duration_ms / 1000):
                    # Speech end detected
                    self.state.is_speech = False
                    event = {'end': self.total_audio_processed}
                    # Reset all states after speech end
                    self.total_audio_processed = 0.0
                    self.state.accumulated_silence = 0
                    self.state.speech_start_time = 0.0
                    self.state.potential_speech_start = 0.0
                    self.state.speech_duration = 0.0

        return event

    def process_audio_chunk(self, audio_chunk: bytes) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Process an audio chunk and determine if it contains speech
        Returns: (should_buffer, speech_dict)
        """
        # Convert and append audio
        audio_float = self._convert_audio(audio_chunk)
        if len(audio_float) == 0:
            return False, None

        # Add to audio buffer for potential prefix padding
        for sample in audio_float:
            self.audio_buffer.append(sample)

        self.current_chunk = np.concatenate([self.current_chunk, audio_float])

        # Track if we should buffer this chunk
        should_buffer = False
        latest_event = None

        # Process complete windows
        window_time = 0
        while len(self.current_chunk) >= self.window_size_samples:
            # Extract window
            window = self.current_chunk[:self.window_size_samples]
            self.current_chunk = self.current_chunk[self.window_size_samples:]

            # Convert to tensor and get speech probability
            window_tensor = torch.from_numpy(window)
            speech_prob = self.model(window_tensor, self.sampling_rate).item()

            # Update window time
            window_time += self.window_size_samples / self.sampling_rate

            # Check if this window contains speech
            is_speech = speech_prob >= self.threshold

            # Buffer audio if we detect potential speech or confirmed speech
            if is_speech or self.state.potential_speech_start > 0 or self.state.is_speech:
                should_buffer = True

            # Update state and get any events
            event = self._update_speech_state(is_speech, window_time)
            if event is not None:
                if 'start' in event:
                    # Get prefix padding audio when speech starts
                    prefix_audio = self._get_prefix_audio()
                    # Adjust start time to account for prefix padding
                    event['start'] = max(0, event['start'] - (len(prefix_audio) / self.sampling_rate))
                    # Add prefix flag to indicate prefix padding is needed
                    event['prefix_samples'] = len(prefix_audio)
                latest_event = event
                logger.debug(f"Speech event detected: {event}")

        return should_buffer, latest_event