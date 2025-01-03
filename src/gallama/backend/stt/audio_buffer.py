from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class VADEvent:
    event_type: str  # "speech_start" or "speech_end"
    timestamp_ms: float
    confidence: float


class AudioBufferWithTiming:
    """Manages audio data with precise timing information."""

    def __init__(self, sample_rate: int):
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = sample_rate
        self.total_samples = 0
        self.start_offset = 0  # Track offset after trimming
        self.last_processed_sample = 0  # Track the last processed sample position
        self.is_processing = False  # Flag to track if processing is ongoing

    def __len__(self) -> int:
        """Return the length of the underlying audio buffer."""
        return len(self.buffer)

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add a new audio chunk to the buffer."""
        self.buffer = np.append(self.buffer, chunk)
        self.total_samples += len(chunk)

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
            # Update last_processed_sample relative to the new buffer position
            self.last_processed_sample = max(0, self.last_processed_sample - sample_index)

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
        Simply moves the last_processed_sample pointer forward to current buffer position.

        Args:
            is_final (bool): Whether this was the final processing of the audio stream
        """
        if is_final:
            self.last_processed_sample = len(self.buffer)
        else:
            # Move the pointer to end of current buffer
            # This ensures we process new chunks from where we left off
            self.last_processed_sample = len(self.buffer)

        self.is_processing = False

    def get_unprocessed_audio(self) -> np.ndarray:
        """
        Get the portion of audio that hasn't been processed yet.
        Returns the audio from last_processed_sample to the end of the buffer.
        """
        # Ensure we don't return empty array and maintain continuity
        if self.last_processed_sample >= len(self.buffer):
            return np.array([], dtype=np.float32)

        return self.buffer[self.last_processed_sample:]

    def reset(self) -> None:
        """Reset the buffer state."""
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples = 0
        self.start_offset = 0
        self.last_processed_sample = 0
        self.is_processing = False