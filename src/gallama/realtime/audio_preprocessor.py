from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class PreprocessingConfig:
    """Configuration for audio preprocessing features"""
    enable_highpass: bool = True
    enable_compression: bool = True
    highpass_cutoff: float = 100  # Cutoff frequency in Hz
    compression_threshold: float = -20  # dB
    compression_ratio: float = 4.0
    attack_time: float = 0.005  # seconds
    release_time: float = 0.1  # seconds
    makeup_gain: float = 6.0  # dB


class AudioPreprocessor:
    def __init__(self, config: PreprocessingConfig, sample_rate: int = 24000):
        """
        Initialize audio preprocessor for float array processing.

        Args:
            config: PreprocessingConfig object containing processing parameters
            sample_rate: Input audio sample rate in Hz (default 24000)
                Note: This should match the input audio sample rate, not the VAD rate
        """
        self.config = config
        self.sample_rate = sample_rate

        # Initialize high-pass filter if enabled
        if self.config.enable_highpass:
            nyquist = self.sample_rate / 2
            normalized_cutoff = self.config.highpass_cutoff / nyquist
            self.hp_b, self.hp_a = signal.butter(4, normalized_cutoff, btype='high')
            self.hp_zi = None  # Filter state for high-pass

        # Initialize compressor parameters if enabled
        if self.config.enable_compression:
            # Convert from dB to linear
            self.threshold = 10 ** (self.config.compression_threshold / 20)
            self.makeup_gain = 10 ** (self.config.makeup_gain / 20)

            # Time constants
            self.attack = np.exp(-1 / (self.sample_rate * self.config.attack_time))
            self.release = np.exp(-1 / (self.sample_rate * self.config.release_time))

            # Envelope follower state
            self.env = 0.0

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to audio.

        Args:
            audio: Input audio as float32 numpy array in range [-1.0, 1.0]

        Returns:
            Filtered audio as float32 numpy array
        """
        if self.hp_zi is None:
            # Initialize filter state
            self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)

        # Apply filter and update state
        filtered, self.hp_zi = signal.lfilter(
            self.hp_b, self.hp_a, audio, zi=self.hp_zi * audio[0]
        )
        return filtered

    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression.

        Args:
            audio: Input audio as float32 numpy array in range [-1.0, 1.0]

        Returns:
            Compressed audio as float32 numpy array
        """
        output = np.zeros_like(audio)
        env = self.env

        for i, sample in enumerate(audio):
            # Envelope detection
            level = abs(sample)
            if level > env:
                env = self.attack * env + (1 - self.attack) * level
            else:
                env = self.release * env + (1 - self.release) * level

            # Gain computation
            if env > self.threshold:
                gain_reduction = self.threshold + (env - self.threshold) / self.config.compression_ratio
                gain = gain_reduction / env if env > 0 else 1.0
            else:
                gain = 1.0

            # Apply gain and makeup gain
            output[i] = sample * gain * self.makeup_gain

        # Update envelope state
        self.env = env
        return output

    def process_float_chunk(self, audio_float: np.ndarray) -> np.ndarray:
        """
        Process an audio chunk in float format.

        Args:
            audio_float: Input audio as float32 numpy array in range [-1.0, 1.0]
                        at input_sample_rate (24000 Hz by default)

        Returns:
            Processed audio as float32 numpy array in range [-1.0, 1.0]
            at the same sample rate as input
        """
        if audio_float.size == 0:
            return audio_float

        try:
            processed_audio = audio_float

            if self.config.enable_highpass:
                processed_audio = self._apply_highpass(processed_audio)

            if self.config.enable_compression:
                processed_audio = self._apply_compression(processed_audio)

            # Ensure output is within [-1, 1] range
            return np.clip(processed_audio, -1.0, 1.0)

        except Exception as e:
            print(f"Error in audio preprocessing: {str(e)}")
            return audio_float

    def reset(self):
        """Reset all internal states"""
        if self.config.enable_highpass:
            self.hp_zi = None
        if self.config.enable_compression:
            self.env = 0.0