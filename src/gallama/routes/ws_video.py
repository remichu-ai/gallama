from fastapi import WebSocket, APIRouter
from ..data_classes import VideoFrame
import struct
from ..dependencies import get_video_collection
from ..logger import logger

video_frames = get_video_collection()

router = APIRouter(prefix="", tags=["video"])

@router.websocket("/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            timestamp = None
            logger.info(f"Received {len(data)} bytes")
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
            video_frames.add_frame(frame)

            # Optional: Send an acknowledgment back to the client
            await websocket.send_text(f"Frame received with timestamp: {frame.timestamp}")
    except Exception as e:
        print(f"WebSocket closed with exception: {e}")
    finally:
        # Check if the WebSocket is still open before closing it
        try:
            # Attempt to close the WebSocket connection gracefully
            await websocket.close()
        except RuntimeError as e:
            # Ignore the error if the WebSocket is already closed
            print(f"WebSocket already closed: {e}")
