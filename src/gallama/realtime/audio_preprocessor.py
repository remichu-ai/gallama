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
    attack_time: float = 0.005
    release_time: float = 0.1
    makeup_gain: float = 6.0


class AudioPreprocessor:
    def __init__(self, config: PreprocessingConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate

        # Initialize high-pass filter
        if self.config.enable_highpass:
            # Design high-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = self.config.highpass_cutoff / nyquist
            self.hp_b, self.hp_a = signal.butter(4, normalized_cutoff, btype='high')
            self.hp_zi = None  # Filter state for high-pass

        # Initialize compressor parameters
        if self.config.enable_compression:
            # Convert from dB to linear
            self.threshold = 10 ** (self.config.compression_threshold / 20)
            self.makeup_gain = 10 ** (self.config.makeup_gain / 20)

            # Time constants
            self.attack = np.exp(-1 / (self.sample_rate * self.config.attack_time))
            self.release = np.exp(-1 / (self.sample_rate * self.config.release_time))

            # State variables
            self.env = 0.0  # Envelope follower state

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to audio"""
        if self.hp_zi is None:
            # Initialize filter state
            self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)

        # Apply filter and update state
        filtered, self.hp_zi = signal.lfilter(
            self.hp_b, self.hp_a, audio, zi=self.hp_zi * audio[0]
        )
        return filtered

    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
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

        # Update state
        self.env = env
        return output

    def process_chunk(self, audio_chunk: bytes) -> bytes:
        """Process an audio chunk with enabled features"""
        try:
            # Convert bytes to float array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Apply enabled processing
            processed_audio = audio_float

            if self.config.enable_highpass:
                processed_audio = self._apply_highpass(processed_audio)

            if self.config.enable_compression:
                processed_audio = self._apply_compression(processed_audio)

            # Ensure output is within [-1, 1] range
            processed_audio = np.clip(processed_audio, -1.0, 1.0)

            # Convert back to int16 bytes
            processed_int16 = (processed_audio * 32768.0).astype(np.int16)
            return processed_int16.tobytes()

        except Exception as e:
            print(f"Error in audio preprocessing: {str(e)}")
            return audio_chunk

    def reset(self):
        """Reset all internal states"""
        if self.config.enable_highpass:
            self.hp_zi = None
        if self.config.enable_compression:
            self.env = 0.0