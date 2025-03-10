from typing import Optional, Deque, List, Union
import time
from PIL import Image
import io
from io import BytesIO
import base64
from collections import deque
from ..logger import logger


# Define the maximum total pixels for each setting
MAX_TOTAL_PIXELS = {
    "240p": 426 * 240,      # ~102K pixels
    "360p": 640 * 360,      # ~230K pixels
    "480p": 854 * 480,      # ~410K pixels
    "540p": 960 * 540,      # ~518K pixels
    "720p": 1280 * 720,     # ~921K pixels
    "900p": 1600 * 900,     # ~1.44M pixels
    "1080p": 1920 * 1080,   # ~2.07M pixels
}


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
        self.last_extraction_time: float = 0

    def add_frame(self, frame: VideoFrame, video_max_resolution: Optional[str] = None):
        """
        Add a frame to the collection. If the collection is full, the oldest frame is removed.
        The frame is rescaled based on total pixel count while maintaining aspect ratio.
        The width and height are adjusted to be multiples of 28.

        Args:
            frame (VideoFrame): The frame to add.
            video_max_resolution (Optional[str]): The maximum resolution setting (e.g., "720p").
        """
        if video_max_resolution and video_max_resolution in MAX_TOTAL_PIXELS:
            max_pixels = MAX_TOTAL_PIXELS[video_max_resolution]
            image = frame.get_image()
            width, height = image.size
            current_pixels = width * height

            if current_pixels > max_pixels:
                # Calculate the scaling ratio based on total pixels
                ratio = (max_pixels / current_pixels) ** 0.5  # Square root to apply evenly to both dimensions
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                # Adjust the width and height to be multiples of 28
                new_width = (new_width // 28) * 28
                new_height = (new_height // 28) * 28

                # Ensure that the dimensions are at least 28
                new_width = max(new_width, 28)
                new_height = max(new_height, 28)

                logger.info(f"Rescaling to max resolution {video_max_resolution} "
                            f"(from {width}x{height} ({current_pixels:,} px) "
                            f"to {new_width}x{new_height} ({new_width * new_height:,} px))")

                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                frame.image = resized_image

        self.frames.append(frame)

    def get_frames_between_timestamps(self, start_time: float, end_time: float) -> List[VideoFrame]:
        """
        Retrieve frames between the specified timestamps.
        Also update the extraction tracker to the latest extracted frame timestamp.
        """
        frames_in_range = [frame for frame in self.frames if start_time <= frame.timestamp <= end_time]
        if frames_in_range:
            # Update the tracker to the timestamp of the last extracted frame.
            self.last_extraction_time = frames_in_range[-1].timestamp
        return frames_in_range

    def get_all_frames(self, start_time: float = 0, only_since_last: bool = True) -> List[VideoFrame]:
        """
        Retrieve all frames in the collection.

        Args:
            only_since_last (bool): If True, returns only frames with a timestamp greater
                                    than the last extraction time. Otherwise, returns all frames.

        Also updates the tracker to the timestamp of the last extracted frame.
        """

        start_time_to_use = max(
            start_time,
            self.last_extraction_time if only_since_last else 0
        )
        logger.info(f"start time is {start_time_to_use}")

        if start_time_to_use:
            frames_to_return = [frame for frame in self.frames if frame.timestamp > start_time_to_use]
        else:
            frames_to_return = list(self.frames)

        if frames_to_return:
            # Update the tracker to the last frame's timestamp from the returned list.
            self.last_extraction_time = frames_to_return[-1].timestamp
        return frames_to_return

    def clear(self):
        """Clear all frames from the collection."""
        self.frames.clear()

    @classmethod
    def get_retained_frames_from_list(
        cls,
        frames: List['VideoFrame'],
        retained_video_frames_per_message: int,
        return_base64: bool = False,
    ) -> Union[List['VideoFrame'], List[str]]:
        """
        Select a number of frames from the provided list and return them.
        If return_base64 is True, converts each frame's image to a base64 Data URI
        (using PNG format), otherwise returns the VideoFrame objects directly.
        When only one frame is requested, the middle frame is selected.
        """
        if not frames:
            return "" if return_base64 else []

        total_frames = len(frames)
        if retained_video_frames_per_message >= total_frames:
            sampled_frames = frames
        else:
            if retained_video_frames_per_message == 1:
                # Select the middle frame
                middle_index = total_frames // 2
                sampled_frames = [frames[middle_index]]
            else:
                # Evenly sample indices from 0 to total_frames-1.
                indices = [
                    int(round(i * (total_frames - 1) / (retained_video_frames_per_message - 1)))
                    for i in range(retained_video_frames_per_message)
                ]
                sampled_frames = [frames[idx] for idx in indices]

        if not return_base64:
            return sampled_frames

        # Convert each sampled frame to a base64 encoded PNG Data URI

        return cls.convert_frames_to_base64(sampled_frames)

    @classmethod
    def sample_frames_time_based(cls, frames: List[VideoFrame], interval: int) -> List[VideoFrame]:
        """
        Retain one frame every `interval` seconds and always include the last frame.
        Assumes frames are in chronological order.
        """
        if not frames:
            return []
        selected = []
        # Start with the first frame
        last_selected_time = frames[0].timestamp
        selected.append(frames[0])
        for frame in frames[1:]:
            if frame.timestamp - last_selected_time >= interval:
                selected.append(frame)
                last_selected_time = frame.timestamp
        # Always include the last frame if not already selected
        if frames[-1] not in selected:
            selected.append(frames[-1])
        return selected

    @classmethod
    def convert_frames_to_base64(cls, frames: List['VideoFrame']) -> List[str]:
        """
        Convert a list of VideoFrame objects to a list of base64-encoded PNG Data URIs.

        Args:
            frames (List[VideoFrame]): The list of video frames to convert.

        Returns:
            List[str]: A list of Data URIs in the format "data:image/png;base64,<base64_data>".
        """
        base64_frames = []
        for frame in frames:
            buffered = BytesIO()
            frame.get_image().save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            b64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64_encoded}"
            base64_frames.append(data_uri)
        return base64_frames