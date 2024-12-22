from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import asyncio
import soundfile as sf
import time
import io
from typing import AsyncIterator, Optional, Literal
from pydantic import BaseModel, validator
import json
from pathlib import Path
from ..dependencies import get_model_manager
from gallama.logger.logger import logger

router = APIRouter(prefix="", tags=["tts"])


class WSMessageTTS(BaseModel):
    type: Literal["add_text", "text_done", "interrupt"]
    text: Optional[str] = None

    @validator("text")
    def validate_text_for_add_type(cls, v, values):
        if values.get("type") == "add_text" and v is None:
            raise ValueError("text field cannot be None when type is add_text")
        return v


class TTSConnectionState:
    """Class to manage the state of a single TTS connection"""

    def __init__(self):
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.processing_flag: asyncio.Event = asyncio.Event()
        self.accumulated_audio: list = []
        self.accumulated_rate: Optional[int] = None
        self.stream_complete: asyncio.Event = asyncio.Event()
        self.text_done: asyncio.Event = asyncio.Event()

        # Set processing flag to True initially
        self.processing_flag.set()

    def reset(self):
        """Reset all state for new connection"""
        # Clear all queues
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

        # Reset all flags and events
        self.processing_flag.clear()
        self.processing_flag.set()  # Reset to initial state
        self.stream_complete.clear()
        self.text_done.clear()

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
        self.stream_end_signal = "STREAM_END"

    async def start(self):
        """Initialize the connection and start processing tasks"""
        await self.websocket.accept()

        # Create and store tasks
        self.tasks.extend([
            asyncio.create_task(self.process_text_stream()),
            asyncio.create_task(self.process_audio_queue())
        ])

    async def process_text_stream(self):
        """Process the incoming text stream and convert to speech"""
        try:
            async def text_stream() -> AsyncIterator[str]:
                while True:
                    await self.state.processing_flag.wait()
                    try:
                        text = await self.state.text_queue.get()
                        if text is None:  # End of stream marker
                            self.state.text_done.set()
                            break
                        logger.info(f"Received text: {text}")
                        yield text
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            await self.tts_model.text_stream_to_speech_to_queue(
                text_stream=text_stream(),
                queue=self.state.audio_queue,
                stream=True,
                # stream_end_signal=self.stream_end_signal,
            )


        except Exception as e:
            logger.error(f"Error in process_text_stream: {e}")
            await self.state.audio_queue.put(Exception(f"Text processing error: {str(e)}"))

    async def process_audio_queue(self):
        """Process and send audio chunks to the client"""
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(self.state.audio_queue.get(), timeout=1.0)

                    if chunk == ("STREAM_END", None):
                        await self.websocket.send_json({"type": "tts_complete"})
                        self.state.stream_complete.set()
                        break

                    if isinstance(chunk, Exception):
                        await self.websocket.send_text(f"Error: {str(chunk)}")
                        break

                    if chunk:
                        sampling_rate, audio_data = chunk
                        if self.state.accumulated_rate is None:
                            self.state.accumulated_rate = sampling_rate
                        self.state.accumulated_audio.append(audio_data)

                        async with self.send_lock:
                            buffer = io.BytesIO()
                            sf.write(
                                buffer,
                                audio_data,
                                sampling_rate,
                                format=self.response_format,
                                subtype="PCM_16"
                            )
                            buffer.seek(0)
                            await self.websocket.send_bytes(buffer.getvalue())

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            logger.error(f"Error in process_audio_queue: {e}")

    async def handle_message(self, message: WSMessageTTS):
        """Handle incoming WebSocket messages"""
        if message.type == "interrupt":
            await self.clear_processing()
        elif message.type == "add_text" and message.text:
            await self.state.text_queue.put(message.text)
        elif message.type == "text_done":
            await self.state.text_queue.put(None)
            self.state.text_done.set()
            await self.state.stream_complete.wait()

    async def clear_processing(self):
        """Clear all processing states and queues"""
        self.state.processing_flag.clear()
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

        self.state.processing_flag.set()

    async def cleanup(self):
        """Cleanup connection resources"""
        # Clear processing flag first to stop new operations
        self.state.processing_flag.clear()

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait(self.tasks, timeout=5.0)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        # Clear all queues and state
        await self.clear_processing()

        # Reset tasks list
        self.tasks = []


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
            await connection.handle_message(parsed_message)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {e}")
    finally:
        if connection:
            # Ensure proper cleanup
            try:
                await connection.cleanup()
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")