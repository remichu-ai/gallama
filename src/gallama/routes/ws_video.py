from fastapi import WebSocket, APIRouter, BackgroundTasks
from ..data_classes import VideoFrame
from typing import Dict
import struct
import asyncio
from ..dependencies import get_video_collection, get_session_config
from ..logger import logger
from pydantic import BaseModel
import os
import time

try:
    from livekit import rtc, api
    from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
except ModuleNotFoundError:
    rtc = None
    api = None
    encode = None
    EncodeOptions = None
    ResizeOptions = None

video_frames = get_video_collection()

router = APIRouter(prefix="", tags=["video"])

@router.websocket("/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    websocket._ping_interval = 60  # Send a ping every 60 seconds
    websocket._ping_timeout = 30   # Wait 30 seconds for a pong response
    try:
        while True:
            data = await websocket.receive_bytes()
            timestamp = None
            logger.info(f"Received video frame")

            session_config = await get_session_config()

            # Check if the message includes a timestamp (first 8 bytes)
            if len(data) >= 8:
                try:
                    # Unpack the first 8 bytes as a double (timestamp)
                    timestamp = struct.unpack('>d', data[:8])[0]
                    frame_data = data[8:]  # The rest is the frame data
                except struct.error:
                    # If unpacking fails, treat the entire message as frame data
                    frame_data = data
            else:
                # If the message is less than 8 bytes, treat it as frame data only
                frame_data = data

            # Create a VideoFrame object
            frame = VideoFrame(frame_data, timestamp)
            video_frames.add_frame(frame, video_max_resolution=session_config.video.video_max_resolution)

            # Optional: Send an acknowledgment back to the client
            await websocket.send_text(f"Frame received with timestamp: {frame.timestamp}")
    except Exception as e:
        logger.info(f"WebSocket closed with exception: {e}")
    finally:
        # Check if the WebSocket is still open before closing it
        try:
            # Attempt to close the WebSocket connection gracefully
            await websocket.close()
        except RuntimeError as e:
            # Ignore the error if the WebSocket is already closed
            logger.error(f"WebSocket already closed: {e}")


class LivekitToken(BaseModel):
    livekit_url: str
    token: str
    save_frame: bool = False

# Global dictionary to store active tasks
active_tasks: Dict[str, asyncio.Task] = {}

@router.post("/v1/video_via_livekit")
async def video_via_livekit(
    token_info: LivekitToken, background_tasks: BackgroundTasks
):
    """Subscribe to the LiveKit room directly and extract one frame per second."""
    livekit_url = token_info.livekit_url
    token = token_info.token
    save_frame = token_info.save_frame

    # Cancel any existing task for this endpoint
    task_key = "video_via_livekit_task"
    if task_key in active_tasks:
        active_tasks[task_key].cancel()
        try:
            await active_tasks[task_key]  # Wait for the task to be cancelled
        except asyncio.CancelledError:
            pass  # Task was successfully cancelled
        del active_tasks[task_key]

    # Start a new background task
    new_task = asyncio.create_task(
        process_video_stream(livekit_url, token, save_frame)
    )
    active_tasks[task_key] = new_task
    background_tasks.add_task(new_task)

    return {"message": "Video processing started"}

async def process_video_stream(livekit_url: str, token: str, save_frame: bool = False):
    """Background task to process the video stream."""
    room = rtc.Room()
    try:
        await room.connect(livekit_url, token)
        session_config = await get_session_config()

        if save_frame:
            save_directory = "/home/remichu/work/ML/gallama/experiment/video_frames"
            os.makedirs(save_directory, exist_ok=True)

        frames_per_second = 1
        frame_interval = 1.0 / frames_per_second
        last_frame_time = 0.0

        async def process_track(track: rtc.RemoteVideoTrack):
            nonlocal last_frame_time
            video_stream = rtc.VideoStream(track)

            async for event in video_stream:
                try:
                    timestamp_us = event.timestamp_us
                    current_time = timestamp_us / 1e6

                    if current_time - last_frame_time >= frame_interval:
                        frame = event.frame
                        image_bytes = encode(
                            frame,
                            EncodeOptions(
                                format="PNG",
                                resize_options=ResizeOptions(
                                    width=480,
                                    height=640,
                                    strategy="scale_aspect_fit",
                                ),
                            ),
                        )

                        if save_frame:
                            image_filename = f"{save_directory}/frame_{timestamp_us}.png"
                            with open(image_filename, "wb") as image_file:
                                image_file.write(image_bytes)

                        frame_obj = VideoFrame(
                            image_bytes,
                            timestamp=time.time()   # use time.time() cause timestamp from livekit is relative timing
                        )
                        video_frames.add_frame(
                            frame_obj,
                            video_max_resolution=session_config.video.video_max_resolution
                        )
                        logger.info(f"Processed frame {current_time}")
                        last_frame_time = current_time
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")

        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):

            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(process_track(track))

        # @room.on("participant_connected")
        # def on_participant_connected(participant: rtc.RemoteParticipant):
        #     logger.info(f"Participant connected: {participant.identity}")

        # Handle existing participants
        for participant in room.remote_participants.values():
            logger.info(f"Found existing participant: {participant.identity}")
            for publication in participant.track_publications.values():
                if (publication.track and
                    (publication.source == rtc.TrackSource.SOURCE_CAMERA
                    or publication.source == rtc.TrackSource.SOURCE_SCREENSHARE)
                ):
                    logger.info(f"Track subscribed: {publication.sid}")
                    asyncio.create_task(process_track(publication.track))

        while True:
            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        logger.info("Video processing task cancelled")
        await room.disconnect()
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        await room.disconnect()
        raise