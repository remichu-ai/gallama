from typing import Optional, Deque, List
import time
from PIL import Image
import io
from collections import deque


class VideoFrame:
    def __init__(self, frame_data: bytes, timestamp: Optional[float] = None):
        """
        Initialize a VideoFrame object.

        Args:
            frame_data (bytes): The raw frame data as bytes.
            timestamp (Optional[float]): The timestamp of the frame. If not provided, the current time is used.
        """
        self.timestamp = timestamp if timestamp is not None else time.time()

        # Convert the frame data (bytes) into a PIL.Image object
        self.image = self._bytes_to_image(frame_data)

    def _bytes_to_image(self, frame_data: bytes) -> Image.Image:
        """
        Convert raw frame data (bytes) into a PIL.Image object.

        Args:
            frame_data (bytes): The raw frame data as bytes.

        Returns:
            Image.Image: The converted PIL.Image object.
        """
        try:
            # Use BytesIO to convert bytes into a file-like object
            image_stream = io.BytesIO(frame_data)
            image = Image.open(image_stream)
            return image
        except Exception as e:
            raise ValueError(f"Failed to convert frame data to PIL.Image: {e}")

    def get_image(self) -> Image.Image:
        """
        Get the PIL.Image object representing the frame.

        Returns:
            Image.Image: The PIL.Image object.
        """
        return self.image

    def get_timestamp(self) -> float:
        """
        Get the timestamp of the frame.

        Returns:
            float: The timestamp of the frame.
        """
        return self.timestamp


class VideoFrameCollection:
    def __init__(self, max_frames: int = 1000):
        self.max_frames = max_frames
        self.frames: Deque[VideoFrame] = deque(maxlen=max_frames)

    def add_frame(self, frame: VideoFrame):
        """Add a frame to the collection. If the collection is full, the oldest frame is removed."""
        self.frames.append(frame)

    def get_frames_between_timestamps(self, start_time: float, end_time: float) -> List[VideoFrame]:
        """Retrieve frames between the specified timestamps."""
        return [frame for frame in self.frames if start_time <= frame.timestamp <= end_time]

    def get_all_frames(self) -> List[VideoFrame]:
        """Retrieve all frames in the collection."""
        return list(self.frames)

    def clear(self):
        """Clear all frames from the collection."""
        self.frames.clear()
