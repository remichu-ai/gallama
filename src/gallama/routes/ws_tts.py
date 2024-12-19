from fastapi import WebSocket, WebSocketDisconnect, Query, APIRouter
import asyncio
import numpy as np
import soundfile as sf
import time
import io
from typing import AsyncIterator, Optional, Literal
from ..dependencies import get_model_manager
import json
from pydantic import BaseModel, validator
from gallama.logger.logger import logger
from pathlib import Path

router = APIRouter(prefix="", tags=["tts"])


class TTSWebsocketManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, dict] = {}
        self.tasks: dict[WebSocket, list[asyncio.Task]] = {}
        self.output_dir = Path("/home/remichu/work/ML/gallama/experiment")
        self.output_dir.mkdir(exist_ok=True)
        self.send_lock = asyncio.Lock()


    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        timestamp = int(time.time())
        self.active_connections[websocket] = {
            'audio_queue': asyncio.Queue(),
            'text_queue': asyncio.Queue(),
            'processing_flag': asyncio.Event(),
            'accumulated_audio': [],  # Store audio chunks
            'accumulated_rate': None,  # Store sampling rate
            'output_file': self.output_dir / f"tts_output_{timestamp}.mp3"  # Unique filename
        }
        self.active_connections[websocket]['processing_flag'].set()  # Start in active state
        self.tasks[websocket] = []

    def disconnect(self, websocket: WebSocket):
        if websocket in self.tasks:
            for task in self.tasks[websocket]:
                task.cancel()
            # Save accumulated audio before disconnecting
            self._save_accumulated_audio(websocket)
        self.tasks.pop(websocket, None)
        self.active_connections.pop(websocket, None)

    async def process_text_stream(self, websocket: WebSocket, tts_model, response_format: str = "mp3"):
        try:
            conn = self.active_connections[websocket]
            # logger.info("Starting text stream processing in TTSWebsocketManager")

            async def text_stream() -> AsyncIterator[str]:
                # logger.info("Initializing text stream iterator")
                while True:
                    # Check if processing is allowed
                    await conn['processing_flag'].wait()

                    try:
                        text = await conn['text_queue'].get()
                        logger.info(f"Got text from queue: {text[:50] if text else 'None'}...")
                        if text is None:  # End of stream marker
                            break
                        yield text
                    except asyncio.CancelledError:
                        break

            # logger.info("Starting TTS model processing")
            await tts_model.text_stream_to_speech_to_queue(
                text_stream=text_stream(),
                queue=conn['audio_queue'],
                stream=True
            )

        except Exception as e:
            print(f"Error in process_text_stream: {e}")
            if websocket in self.active_connections:
                await conn['audio_queue'].put(Exception(f"Text processing error: {str(e)}"))

    def _save_accumulated_audio(self, websocket: WebSocket):
        """Save accumulated audio chunks to file"""
        conn = self.active_connections.get(websocket)
        if conn and conn['accumulated_audio'] and conn['accumulated_rate']:
            # Concatenate all audio chunks
            combined_audio = np.concatenate(conn['accumulated_audio'])
            # Save to file
            sf.write(
                conn['output_file'],
                combined_audio,
                conn['accumulated_rate'],
                format=str(conn['output_file'].suffix[1:])  # Remove dot from suffix
            )
            print(f"Saved audio to: {conn['output_file']}")

    async def clear_processing(self, websocket: WebSocket):
        """Clear current processing and buffers for a connection"""
        if websocket in self.active_connections:
            conn = self.active_connections[websocket]
            # Stop current processing
            conn['processing_flag'].clear()

            # Save current accumulated audio before clearing
            self._save_accumulated_audio(websocket)

            # Reset accumulation
            conn['accumulated_audio'] = []
            conn['accumulated_rate'] = None

            # Create new output file
            timestamp = int(time.time())
            conn['output_file'] = self.output_dir / f"tts_output_{timestamp}.mp3"

            # Clear the queues
            while not conn['text_queue'].empty():
                try:
                    conn['text_queue'].get_nowait()
                except asyncio.QueueEmpty:
                    break

            while not conn['audio_queue'].empty():
                try:
                    conn['audio_queue'].get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Reset processing flag to allow new processing
            conn['processing_flag'].set()

    async def process_audio_queue(self, websocket: WebSocket, response_format: str = "mp3"):
        try:
            conn = self.active_connections[websocket]

            while True:
                # Add timeout to prevent infinite waiting
                try:
                    chunk = await asyncio.wait_for(conn['audio_queue'].get(), timeout=5.0)
                    # if chunk is None:
                    #     # Save final audio before ending
                    #     self._save_accumulated_audio(websocket)
                    #     await websocket.send_bytes(b"END_OF_STREAM")
                    #     break

                    if isinstance(chunk, Exception):
                        await websocket.send_text(f"Error: {str(chunk)}")
                        break

                    # Only process if flag is set and ensure completion
                    #if conn['processing_flag'].is_set():
                    if chunk:
                        sampling_rate, audio_data = chunk
                        logger.info(
                            f"Processing audio chunk - Queue size: {conn['audio_queue'].qsize()}, Audio shape: {audio_data.shape}, Timestamp: {time.time()}")
                        # Add pre-send logging
                        logger.info(f"About to send chunk - Size: {len(audio_data)}, Time: {time.time()}")

                        # Store the chunk for accumulation
                        if conn['accumulated_rate'] is None:
                            conn['accumulated_rate'] = sampling_rate
                        conn['accumulated_audio'].append(audio_data)

                        # In process_audio_queue
                        async with self.send_lock:
                            buffer = io.BytesIO()
                            sf.write(buffer, audio_data, sampling_rate, format=response_format)
                            buffer.seek(0)
                            await websocket.send_bytes(buffer.getvalue())

                    # # Send to client
                    # buffer = io.BytesIO()
                    # sf.write(buffer, audio_data, sampling_rate, format=response_format)
                    # buffer.seek(0)
                    # data_to_send = buffer.getvalue()
                    # logger.info(f"Sending websocket chunk - Bytes: {len(data_to_send)}, Time: {time.time()}")
                    # await websocket.send_bytes(data_to_send)
                    # logger.info(f"Chunk sent successfully - Time: {time.time()}")
                    # Signal completion
                    # conn['audio_queue'].task_done()

                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            print(f"Error in process_audio_queue: {e}")
            # Try to save accumulated audio even if there's an error
            self._save_accumulated_audio(websocket)

class WSMessageTTS(BaseModel):
    type: str = Literal["add_text", "clear"]
    text: Optional[str] = None

    @validator("text")
    def validate_text_for_add_type(cls, v, values):
        if values.get("type") == "add_text" and v is None:
            raise ValueError("text field cannot be None when type is add_text")
        return v

async def validate_message(raw_data: str) -> WSMessageTTS:
    try:
        json_data = json.loads(raw_data)
        return WSMessageTTS(**json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Invalid message format: {str(e)}")

@router.websocket("/ws/speech")
async def websocket_endpoint(
    websocket: WebSocket,
    #model: str = Query(...),
    #response_format: str = Query("mp3", regex="^(mp3|wav|flac)$")
):
    manager = TTSWebsocketManager()
    model_manager = get_model_manager()
    tts = model_manager.tts_dict.get("gpt_sovits")
    response_format = "wav"
    logger.info("TODO here")

    # Check if model exists before accepting connection
    if not tts:
        await websocket.accept()
        error_data = {
            "type": "error",
            "message": f"Model '{tts.model_name}' not found. Available models: {list(model_manager.tts_dict.keys())}"
        }
        await websocket.send_json(error_data)
        await websocket.close()
        return

    try:
        await manager.connect(websocket)
        # logger.info("WebSocket connected successfully")

        # Create an event for signaling stream end
        stream_end_event = asyncio.Event()

        # Start the text-to-speech processing task
        # logger.info("Starting text processing task")
        text_task = asyncio.create_task(
            manager.process_text_stream(websocket, tts, response_format)
        )
        manager.tasks[websocket].append(text_task)

        # Start the audio processing task
        # logger.info("Starting audio processing task")
        audio_task = asyncio.create_task(
            manager.process_audio_queue(websocket, response_format)
        )
        manager.tasks[websocket].append(audio_task)

        try:
            while True:
                message = await websocket.receive_text()
                message = await validate_message(message)
                # logger.info(f"Received message: {message}")
                if message.type == "clear":
                    # logger.info("Clearing processing queue")
                    await manager.clear_processing(websocket)
                elif message.type == "add_text" and message.text != "":
                    # logger.info(f"Adding text to queue: {message.text[:50]}...")
                    await manager.active_connections[websocket]['text_queue'].put(message.text)

        except WebSocketDisconnect:
            print("WebSocket disconnected")

        # Wait for tasks to complete
        if websocket in manager.tasks:
            await asyncio.gather(*manager.tasks[websocket])

    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
    finally:
        manager.disconnect(websocket)