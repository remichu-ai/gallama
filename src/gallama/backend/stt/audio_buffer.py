from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from gallama.logger import logger

@dataclass
class VADEvent:
    event_type: str  # "speech_start" or "speech_end"
    timestamp_ms: float
    confidence: float


class AudioBufferWithTiming:
    """Manages audio data with precise timing information."""

    def __init__(self, sample_rate: int, max_length_minutes: float = 30.0, trim_duration_minutes: float = 5.0):
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = sample_rate
        self.total_samples = 0
        self.start_offset = 0  # Track offset after trimming
        self.last_processed_sample = 0  # Track the last processed sample position for ASR
        self.last_processed_sample_vad = 0  # Track the last processed sample position for VAD
        self.is_processing = False  # Flag to track if processing is ongoing
        self.max_length_minutes = max_length_minutes
        self.trim_duration_minutes = trim_duration_minutes
        self.max_length_samples = int(self.max_length_minutes * 60 * self.sample_rate)
        self.trim_length_samples = int(self.trim_duration_minutes * 60 * self.sample_rate)

    def __len__(self) -> int:
        """Return the length of the underlying audio buffer."""
        return len(self.buffer)

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add a new audio chunk to the buffer."""
        self.buffer = np.append(self.buffer, chunk)
        self.total_samples += len(chunk)

        # Check if the buffer exceeds the maximum length
        if len(self.buffer) > self.max_length_samples:
            # Instead of just removing excess, remove trim_duration worth of samples
            self.buffer = self.buffer[self.trim_length_samples:]
            self.start_offset += self.trim_length_samples
            self.last_processed_sample = max(0, self.last_processed_sample - self.trim_length_samples)
            self.last_processed_sample_vad = max(0, self.last_processed_sample_vad - self.trim_length_samples)

    def get_time_ms(self, sample_index: int) -> float:
        """Convert sample index to milliseconds from stream start."""
        absolute_sample = sample_index + self.start_offset
        return (absolute_sample / self.sample_rate) * 1000

    def get_current_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds."""
        return (len(self.buffer) / self.sample_rate) * 1000

    def clear_until(self, sample_index: int) -> None:
        """Clear buffer up to given sample index, updating timing information."""
        if sample_index > 0:
            self.buffer = self.buffer[sample_index:]
            self.start_offset += sample_index
            # Update both processing trackers relative to the new buffer position
            self.last_processed_sample = max(0, self.last_processed_sample - sample_index)
            self.last_processed_sample_vad = max(0, self.last_processed_sample_vad - sample_index)

    def get_samples_for_duration(self, duration_ms: float) -> Optional[np.ndarray]:
        """Get samples corresponding to specified duration from the start."""
        num_samples = int((duration_ms / 1000) * self.sample_rate)
        if num_samples <= len(self.buffer):
            return self.buffer[:num_samples]
        return None

    def get_latest_samples(self, duration_ms: float) -> Optional[np.ndarray]:
        """Get the most recent samples of specified duration."""
        num_samples = int((duration_ms / 1000) * self.sample_rate)
        if num_samples <= len(self.buffer):
            return self.buffer[-num_samples:]
        return None

    def mark_processing_start(self) -> None:
        """Mark the start of audio processing."""
        self.is_processing = True

    def mark_processing_complete(self, is_final: bool = False) -> None:
        """
        Mark the completion of audio processing.
        Updates the last_processed_sample pointer to current buffer position.

        Args:
            is_final (bool): Whether this was the final processing of the audio stream
        """
        if is_final:
            self.last_processed_sample = len(self.buffer)
            self.last_processed_sample_vad = len(self.buffer)
        else:
            # Move the pointer to end of current buffer
            # self.last_processed_sample = len(self.buffer)
            pass

        self.is_processing = False

    def get_unprocessed_audio(self) -> np.ndarray:
        """
        Get the portion of audio that hasn't been processed yet by ASR.
        Returns the audio from last_processed_sample to the end of the buffer.
        """
        if self.last_processed_sample >= len(self.buffer):
            return np.array([], dtype=np.float32)

        return self.buffer[self.last_processed_sample:]

    def get_unprocessed_audio_vad(self) -> np.ndarray:
        logger.info(
            f"VAD processing state - last_processed: {self.last_processed_sample_vad}, buffer_len: {len(self.buffer)}, start_offset: {self.start_offset}")
        if self.last_processed_sample_vad >= len(self.buffer):
            return np.array([], dtype=np.float32)
        return self.buffer[self.last_processed_sample_vad:]

    def mark_vad_processing_complete(self, is_final: bool = False) -> None:
        """
        Mark the completion of VAD processing.
        Updates the last_processed_sample_vad pointer to current buffer position.

        Args:
            is_final (bool): Whether this was the final processing of the audio stream
        """
        if is_final:
            self.last_processed_sample_vad = len(self.buffer)
        else:
            self.last_processed_sample_vad = len(self.buffer)

    def reset(self) -> None:
        """Reset the buffer state."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples = 0
        self.start_offset = 0
        self.last_processed_sample = 0
        self.last_processed_sample_vad = 0
        self.is_processing = False