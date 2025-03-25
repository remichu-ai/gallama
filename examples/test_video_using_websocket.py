import asyncio
import websockets
import json
import uuid
from enum import Enum
from typing import List, Optional
import os
import struct
import cv2
import numpy as np
import time

from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="test", websocket_base_url="ws://127.0.0.1:8000/v1")

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview-2024-10-01") as connection:
        await connection.session.update(session={
            'modalities': ['text'],
            'video': {
                'video_stream': True,
                'video_max_resolution': "720p"
                # 'video_max_resolution': "540p"
                # 'video_max_resolution': None
            }
        })

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "what do you see in the video?"}],
                # "content": [{"type": "input_text", "text": "how many scissor do you see?"}],
                # "content": [{"type": "input_text", "text": "do you see any coffee appliances"}],
            }
        )

        frames_per_second = 2  # Change this value as needed
        await send_video_frames(
            video_path="/home/remichu/work/ML/gallama/experiment/media/kitchen.mp4",
            frames_per_second=frames_per_second
        )

        await connection.response.create()

        start = time.time()
        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print("text done")

            elif event.type == "response.done":
                print(f"time taken: {time.time() - start}")
                break


async def send_video_frames(video_path: str, frames_per_second: int = 1):
    # Connect to the video WebSocket server
    async with websockets.connect('ws://localhost:8000/video') as websocket:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)  # Calculate the interval between frames
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frames based on the specified frames_per_second
            if frame_count % frame_interval == 0:
                # Convert the frame to PNG format
                _, buffer = cv2.imencode('.png', frame)
                frame_data = buffer.tobytes()

                # Pack the frame data with a timestamp (optional)
                timestamp = frame_count / fps  # Calculate the timestamp
                packed_data = struct.pack('>d', timestamp) + frame_data

                # Send the frame data to the WebSocket server
                await websocket.send(packed_data)
                print(f"Sent frame at {timestamp:.2f} seconds")

            frame_count += 1

        cap.release()
        print("Finished sending video frames.")

if __name__ == "__main__":
    print("Starting WebSocket LLM and video test...")
    asyncio.run(main())