from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import asyncio
import soundfile as sf
import time
import io
from typing import AsyncIterator, Optional, Literal
from ..data_classes import TTSEvent, WSMessageTTS
import samplerate
import numpy as np
import json
from pathlib import Path
from ..dependencies import get_model_manager
from gallama.logger.logger import logger
router = APIRouter(prefix="", tags=["tts"])




class TTSConnectionState:
    """Class to manage the state of a single TTS connection"""

    def __init__(self):
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.text_queue: asyncio.Queue = asyncio.Queue()
        # self.processing_flag: asyncio.Event = asyncio.Event()
        self.accumulated_audio: list = []
        self.accumulated_rate: Optional[int] = None
        self.mode: Literal["processing", "idle"] = "idle"
        self.stream_complete: asyncio.Event = asyncio.Event()
        self.text_done: asyncio.Event = asyncio.Event()
        self.last_text_event_time: float = 0.0
        self.timeout_seconds: int = 12

        self.lock = asyncio.Lock()

    def reset(self):
        """Reset all state for new connection"""
        # Clear all queues
        self.mode = "idle"
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset accumulated data
        self.accumulated_audio = []
        self.accumulated_rate = None



class TTSConnection:
    """Class to handle a single TTS WebSocket connection"""

    def __init__(self, websocket: WebSocket, tts_model, response_format: str = "wav"):
        self.websocket = websocket
        self.tts_model = tts_model
        self.response_format = response_format
        self.state = TTSConnectionState()
        self.tasks: list[asyncio.Task] = []
        self.send_lock = asyncio.Lock()
        self.target_sample_rate = 24000

    async def start(self):
        """Initialize the connection and start processing tasks"""
        await self.websocket.accept()

    async def process_audio_queue(self):
        """Process and send audio chunks to the client"""
        try:
            resampler = samplerate.Resampler('sinc_medium', channels=1)
            buffer_threshold = 2048  # Experiment with this value
            accumulated_chunk = np.array([], dtype=np.float32)

            while True:
                try:
                    chunk = await asyncio.wait_for(self.state.audio_queue.get(), timeout=1.0)

                    if chunk is None:
                        logger.info("Received None chunk")
                        continue

                    if isinstance(chunk, TTSEvent) and chunk.type == "text_end":
                        async with self.state.lock:
                            self.state.mode = "idle"
                            break

                    if chunk and self.state.mode != "idle":
                        sampling_rate, audio_data = chunk

                        # Ensure audio data is float32 and normalized to [-1, 1]
                        audio_data = audio_data.astype(np.float32)
                        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                            audio_data = audio_data / np.max(np.abs(audio_data))

                        # Accumulate chunks
                        accumulated_chunk = np.concatenate([accumulated_chunk, audio_data])

                        # Only process when we have enough data
                        if len(accumulated_chunk) >= buffer_threshold:
                            if sampling_rate != self.target_sample_rate:
                                ratio = self.target_sample_rate / sampling_rate
                                resampled_audio = resampler.process(
                                    accumulated_chunk,
                                    ratio,
                                    end_of_input=False
                                )
                            else:
                                resampled_audio = accumulated_chunk

                            # Reset accumulation
                            accumulated_chunk = np.array([], dtype=np.float32)

                            async with self.send_lock:
                                try:
                                    buffer = io.BytesIO()
                                    sf.write(
                                        buffer,
                                        resampled_audio,
                                        self.target_sample_rate,
                                        format=self.response_format,
                                        subtype="PCM_16"
                                    )
                                    buffer.seek(0)
                                    await self.websocket.send_bytes(buffer.getvalue())
                                except RuntimeError as e:
                                    logger.info(f"Websocket send failed: {e}")
                                    break

                except asyncio.TimeoutError:
                    if self.state.mode == "idle":
                        break
                    continue

        except Exception as e:
            logger.error(f"Error in process_audio_queue: {e}")
            raise

    async def track_incoming_text(self):
        while True:
            await asyncio.sleep(0.2)
            if self.state.mode == "processing":
                if time.time() - self.state.last_text_event_time > self.state.timeout_seconds:
                    logger.info("-------------------------------------------text timeout")
                    await self.state.text_queue.put(TTSEvent(type="text_end"))
                    break
            elif self.state.mode == "idle":
                break


    async def handle_message(self, message: WSMessageTTS):
        """Handle incoming WebSocket messages"""
        if message.type == "interrupt":
            pass
            # await self.clear_processing()   # TODO
        elif message.type == "add_text" and message.text:
            if self.state.mode == "idle":
                async with self.state.lock:
                    self.state.reset()
                    logger.info("-------------------------------------------reset state")
                    self.state.mode = "processing"
                    # reset state to ensure empty queue


                await self.state.text_queue.put(TTSEvent(type="text_start"))
                await self.state.text_queue.put(message.text)
                self.state.last_text_event_time = time.time()

                asyncio.create_task(self.track_incoming_text())
                asyncio.create_task(self.tts_model.text_stream_to_speech_to_queue(
                    text_queue=self.state.text_queue,
                    queue=self.state.audio_queue,
                    stream=True,
                ))
                asyncio.create_task(self.process_audio_queue())

            else:
                await self.state.text_queue.put(message.text)
                self.state.last_text_event_time = time.time()
        elif message.type == "text_done":
            await self.state.text_queue.put(TTSEvent(type="text_end"))


    async def clear_processing(self):
        """Clear all processing states and queues"""
        # self.state.processing_flag.clear()
        self.state.accumulated_audio = []
        self.state.accumulated_rate = None

        # Clear all events
        self.state.text_done.clear()
        self.state.stream_complete.clear()

        # Clear queues
        while not self.state.text_queue.empty():
            try:
                self.state.text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self.state.audio_queue.empty():
            try:
                self.state.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def cleanup(self):
        """Cleanup connection resources"""
        try:
            # Stop any ongoing text tracking
            self.state.mode = "idle"

            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Clear all queues
            await self.clear_queues()

            # Reset state
            self.state.reset()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def clear_queues(self):
        """Clear all queues safely"""
        # Clear audio queue
        while not self.state.audio_queue.empty():
            try:
                self.state.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear text queue
        while not self.state.text_queue.empty():
            try:
                self.state.text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


async def validate_message(raw_data: str) -> WSMessageTTS:
    """Validate and parse incoming WebSocket messages"""
    try:
        json_data = json.loads(raw_data)
        return WSMessageTTS(**json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid message format: {str(e)}")


@router.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    model_manager = get_model_manager()
    model_name = "gpt_sovits"
    tts = model_manager.tts_dict.get(model_name)

    if not tts:
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "message": f"Model '{model_name}' not found"
        })
        await websocket.close()
        return

    connection = None
    try:
        # Create new connection
        connection = TTSConnection(websocket, tts)
        await connection.start()

        while True:
            message = await websocket.receive_text()
            parsed_message = await validate_message(message)

            # Track the tasks created by handle_message
            if parsed_message.type == "add_text" and parsed_message.text:
                connection.tasks = []  # Clear previous tasks
                await connection.handle_message(parsed_message)
            else:
                await connection.handle_message(parsed_message)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {e}")
    finally:
        if connection:
            try:
                await connection.cleanup()
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")
            finally:
                try:
                    await websocket.close()
                except RuntimeError:
                    # Websocket might already be closed
                    pass