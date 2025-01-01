from silero_vad import load_silero_vad, VADIterator
from ..data_classes.realtime_client_proto import TurnDetectionConfig
import numpy as np
from typing import Tuple, Optional, Dict
import torch
from dataclasses import dataclass
import samplerate
from collections import deque
from .audio_preprocessor import AudioPreprocessor
import os
import soundfile as sf
from datetime import datetime

from ..dependencies_server import get_server_logger

logger = get_server_logger()


@dataclass
class VADState:
    is_speech: bool = False
    speech_start_time: float = 0.0
    accumulated_silence: float = 0.0
    last_window_time: float = 0.0
    potential_speech_start: float = 0.0
    speech_duration: float = 0.0
    speech_samples: Optional[np.ndarray] = None
    original_speech_samples: Optional[np.ndarray] = None  # Store original sample rate speech


@dataclass
class PreprocessingConfig:
    """Separate configuration for audio preprocessing"""
    enable_highpass: bool = True
    enable_compression: bool = True
    highpass_cutoff: float = 100
    compression_threshold: float = -20
    compression_ratio: float = 3.0
    attack_time: float = 0.005
    release_time: float = 0.1
    makeup_gain: float = 7.0


class VADProcessor:
    def __init__(self, turn_detection_config: TurnDetectionConfig, preprocessing_config: PreprocessingConfig = None,
                 input_sample_rate: int = 24000, debug: bool = True,
                 debug_folder_path: str = "/home/remichu/work/ML/gallama/experiment/log"):
        self.model = load_silero_vad()
        self.model.eval()

        # Sample rate configuration
        self.input_sample_rate = input_sample_rate
        self.vad_sample_rate = 16000
        self.resampler = samplerate.Resampler('sinc_best', channels=1)
        self.resample_ratio = self.vad_sample_rate / self.input_sample_rate

        # Debug settings
        self.debug = debug
        self.debug_folder_path = debug_folder_path
        if self.debug and not os.path.exists(debug_folder_path):
            os.makedirs(debug_folder_path)

        # Initialize audio preprocessor if enabled
        if turn_detection_config.enable_preprocessing:
            if preprocessing_config is None:
                preprocessing_config = PreprocessingConfig()
            self.audio_preprocessor = AudioPreprocessor(preprocessing_config)
        else:
            self.audio_preprocessor = None

        self.threshold = turn_detection_config.threshold
        self.silence_duration_ms = turn_detection_config.silence_duration_ms
        self.min_speech_duration_ms = 250
        self.create_response = turn_detection_config.create_response

        # Fixed parameters
        self.window_size_samples = 512
        self.speech_pad_ms = 70

        # State management
        self.state = VADState()
        self.current_chunk = np.array([], dtype=np.float32)

        # Initialize VAD iterator
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=self.threshold,
            sampling_rate=self.vad_sample_rate,
            min_silence_duration_ms=self.silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms
        )
        self.total_audio_processed = 0.0

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from input sample rate to VAD sample rate"""
        if audio.size == 0:
            return np.array([], dtype=np.float32)
        return self.resampler.process(audio, self.resample_ratio)

    def _save_debug_audio(self, audio_data: np.ndarray):
        """Save debug audio to file"""
        if not self.debug or not self.debug_folder_path:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.debug_folder_path, f"speech_segment_{timestamp}.wav")

        # Save using input sample rate for original quality
        sf.write(filename, audio_data, self.input_sample_rate)
        logger.debug(f"Saved debug audio to {filename}")

    def process_audio_chunk(self, audio_float: np.ndarray) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Process an audio chunk and determine if it contains speech
        Args:
            audio_float: Numpy array of float32 audio data, normalized to [-1.0, 1.0]
        Returns: (should_buffer, speech_dict)
        """
        # Apply preprocessing if enabled
        if self.audio_preprocessor:
            audio_float = self.audio_preprocessor.process_float_chunk(audio_float)

        if audio_float.size == 0:
            return False, None

        # Store original audio for speech segments if debugging
        if self.debug and self.state.is_speech:
            if self.state.original_speech_samples is None:
                self.state.original_speech_samples = audio_float
            else:
                self.state.original_speech_samples = np.concatenate(
                    [self.state.original_speech_samples, audio_float])

        # Resample audio to VAD sample rate
        resampled_audio = self._resample_audio(audio_float)
        self.current_chunk = np.concatenate([self.current_chunk, resampled_audio])

        should_buffer = False
        latest_event = None

        # Process complete windows
        while self.current_chunk.size >= self.window_size_samples:
            window = self.current_chunk[:self.window_size_samples]
            self.current_chunk = self.current_chunk[self.window_size_samples:]

            window_tensor = torch.from_numpy(window)
            speech_prob = self.model(window_tensor, self.vad_sample_rate).item()

            # Update window time using input sample rate for accurate timing
            window_time = self.window_size_samples / self.vad_sample_rate
            self.total_audio_processed += window_time

            is_speech = speech_prob >= self.threshold

            if is_speech or self.state.potential_speech_start > 0 or self.state.is_speech:
                should_buffer = True

            # Update state and get events
            event = self._update_speech_state(is_speech)
            if event is not None:
                latest_event = event

        return should_buffer, latest_event

    def _update_speech_state(self, is_speech: bool) -> Optional[Dict[str, float]]:
        """Update speech state and return speech events"""
        event = None

        if is_speech:
            if not self.state.is_speech:
                if self.state.potential_speech_start == 0.0:
                    self.state.potential_speech_start = self.total_audio_processed
                    # Initialize speech samples when speech potentially starts
                    if self.debug:
                        self.state.original_speech_samples = None

                current_duration = (self.total_audio_processed - self.state.potential_speech_start) * 1000

                if current_duration >= self.min_speech_duration_ms:
                    self.state.is_speech = True
                    self.state.speech_start_time = self.state.potential_speech_start
                    event = {'start': self.state.speech_start_time}
                    logger.debug(f"Speech started after reaching minimum duration: {current_duration:.1f}ms")

        else:
            if not self.state.is_speech:
                self.state.potential_speech_start = 0.0
                if self.debug:
                    self.state.original_speech_samples = None
            else:
                self.state.accumulated_silence += self.window_size_samples / self.vad_sample_rate

                if self.state.accumulated_silence >= (self.silence_duration_ms / 1000):
                    event = {'end': self.total_audio_processed}

                    if self.debug and self.state.original_speech_samples is not None:
                        self._save_debug_audio(self.state.original_speech_samples)

                    # Reset all state
                    self.state.is_speech = False
                    self.total_audio_processed = 0.0
                    self.state.accumulated_silence = 0
                    self.state.speech_start_time = 0.0
                    self.state.potential_speech_start = 0.0
                    self.state.speech_duration = 0.0
                    self.state.original_speech_samples = None

        return event

    def reset(self):
        """Reset all state"""
        if self.audio_preprocessor:
            self.audio_preprocessor.reset()
        self.vad_iterator.reset_states()
        self.current_chunk = np.array([], dtype=np.float32)
        self.state = VADState()
        self.total_audio_processed = 0.0

