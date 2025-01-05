from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import asyncio
import soundfile as sf
import time
import io
from typing import AsyncIterator, Optional, Literal
from ..data_classes import TTSEvent, WSInterTTS, WSInterConfigUpdate, WSInterCancel, WSInterCleanup
import samplerate
import numpy as np
import json
from pathlib import Path

from ..data_classes.realtime_client_proto import SessionConfig
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
        self.session_config = SessionConfig()
        self.websocket = websocket
        self.tts_model = tts_model
        self.response_format = response_format
        self.state = TTSConnectionState()
        self.tasks: list[asyncio.Task] = []
        self.send_lock = asyncio.Lock()

        self._active_tasks: set[asyncio.Task] = set()  # Track active tasks

    async def start(self):
        """Initialize the connection and start processing tasks"""
        await self.websocket.accept()

    async def cancel_processing(self):
        """Cancel all ongoing processing and reset state"""
        logger.info("Canceling TTS processing")

        # Stop the TTS model processing if needed
        if hasattr(self.tts_model, 'stop'):
            self.tts_model.stop()

        # Create a copy of the tasks set to avoid modification during iteration
        tasks_to_cancel = list(self._active_tasks)

        # Cancel all tasks
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if tasks_to_cancel:
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except asyncio.CancelledError:
                pass

        # Clear the active tasks set after all tasks are cancelled
        self._active_tasks.clear()

        # Reset state
        async with self.state.lock:
            self.state.reset()
            self.state.mode = "idle"

        # Clear all queues
        await self.clear_queues()

        # Notify client that processing was cancelled
        try:
            await self.websocket.send_json({
                "type": "tts_cancelled"
            })
        except RuntimeError:
            # Websocket might be closed
            pass


    def create_task(self, coroutine) -> asyncio.Task:
        """Create a new task and add it to active tasks set"""
        task = asyncio.create_task(coroutine)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    async def process_audio_queue(self):
        """Process and send audio chunks to the client"""
        try:
            resampler = samplerate.Resampler('sinc_medium', channels=1)
            buffer_threshold = 2048  # Buffer size for processing
            accumulated_chunk = np.array([], dtype=np.float32)
            prev_chunk_end = np.zeros(256, dtype=np.float32)  # Store end of previous chunk for crossfade
            is_first_chunk = True  # Flag to track first chunk for initial fade-in

            def apply_crossfade(current_chunk: np.ndarray, prev_end: np.ndarray, fade_length: int = 256) -> np.ndarray:
                """Apply crossfade between chunks to prevent popping"""
                if len(current_chunk) < fade_length:
                    return current_chunk

                # Create fade curves
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)

                # Apply crossfade at the beginning
                current_chunk[:fade_length] = (current_chunk[:fade_length] * fade_in) + (prev_end * fade_out)
                return current_chunk

            def apply_initial_fade_in(audio_data: np.ndarray, fade_length: int = 512) -> np.ndarray:
                """Apply a longer, smoother fade-in for the very first chunk"""
                if len(audio_data) < fade_length:
                    fade_length = len(audio_data)

                # Create smooth fade-in curve using half of a Hann window
                fade_curve = np.hanning(fade_length * 2)[:fade_length]

                # Apply fade-in
                audio_data = audio_data.copy()  # Create a copy to prevent modifying original
                audio_data[:fade_length] *= fade_curve
                return audio_data

            def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
                """
                Normalize audio to match OpenAI's range (~0.45 peak amplitude)
                """
                # Remove DC offset first as you were doing
                audio_data = audio_data - np.mean(audio_data)

                # Target peak of 0.45 to match OpenAI's observed range
                target_peak = 0.45

                # Calculate current peak
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:  # Prevent division by zero
                    audio_data = audio_data * (target_peak / max_val)

                # Add safety clip to ensure we're within PCM_16 valid range
                audio_data = np.clip(audio_data, -0.99, 0.99)

                return audio_data

            def process_chunk(audio_data: np.ndarray, is_first: bool = False) -> np.ndarray:
                """Process a single chunk of audio data"""
                # First normalize the audio
                audio_data = normalize_audio(audio_data)

                # Apply initial fade-in only to the first chunk
                if is_first:
                    audio_data = apply_initial_fade_in(audio_data)

                return audio_data

            while True:
                try:
                    chunk = await asyncio.wait_for(self.state.audio_queue.get(), timeout=1.0)

                    if chunk is None:
                        logger.info("Received None chunk")
                        continue

                    if isinstance(chunk, TTSEvent) and chunk.type == "text_end":
                        logger.info("TTS Done")
                        async with self.state.lock:
                            self.state.mode = "idle"

                        # send client response that TTS completed
                        await self.websocket.send_json({
                            "type": "tts_complete"
                        })
                        break

                    if chunk and self.state.mode != "idle":
                        sampling_rate, audio_data = chunk

                        # Convert to float32
                        audio_data = audio_data.astype(np.float32)

                        # First normalization pass with initial fade-in if it's the first chunk
                        audio_data = process_chunk(audio_data, is_first_chunk)

                        # Accumulate chunks
                        accumulated_chunk = np.concatenate([accumulated_chunk, audio_data])

                        # Only process when we have enough data
                        if len(accumulated_chunk) >= buffer_threshold:
                            # Resample if needed
                            if sampling_rate != self.session_config.output_sample_rate:
                                ratio = self.session_config.output_sample_rate / sampling_rate
                                resampled_audio = resampler.process(
                                    accumulated_chunk,
                                    ratio,
                                    end_of_input=False
                                )
                            else:
                                resampled_audio = accumulated_chunk

                            # Second normalization pass after resampling
                            resampled_audio = normalize_audio(resampled_audio)

                            # Apply crossfade with previous chunk
                            if not is_first_chunk:  # Skip crossfade for the first chunk
                                resampled_audio = apply_crossfade(resampled_audio, prev_chunk_end)

                            # Save end of current chunk for next crossfade
                            prev_chunk_end = resampled_audio[-256:].copy()

                            # Reset accumulation and update first chunk flag
                            accumulated_chunk = np.array([], dtype=np.float32)
                            is_first_chunk = False

                            # async with self.send_lock:
                            #     try:
                            #         buffer = io.BytesIO()
                            #         # Convert to int16 for sending
                            #         sf.write(
                            #             buffer,
                            #             resampled_audio,
                            #             self.target_sample_rate,
                            #             format=self.response_format,
                            #             subtype="PCM_16"
                            #         )
                            #         buffer.seek(0)
                            #         await self.websocket.send_bytes(buffer.getvalue())
                            #     except RuntimeError as e:
                            #         logger.info(f"Websocket send failed: {e}")
                            #         break

                            async with self.send_lock:
                                try:
                                    # Since resampled_audio is already normalized to [-0.99, 0.99] from your normalize_audio function,
                                    # and the worklet will divide by 32768, we need to multiply by 32768 here
                                    int16_data = (resampled_audio * 32768).astype(np.int16)
                                    raw_bytes = int16_data.tobytes()
                                    await self.websocket.send_bytes(raw_bytes)
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


    async def handle_message(self, message: WSInterTTS | WSInterConfigUpdate | WSInterCancel | WSInterCleanup):
        """Handle incoming WebSocket messages"""
        if message.type == "common.cancel" or message.type == "common.cleanup" or message.type == "common.cleanup":
            await self.cancel_processing()
        elif message.type == "tts.add_text" and message.text:
            if self.state.mode == "idle":
                async with self.state.lock:
                    self.state.reset()
                    self.state.mode = "processing"

                await self.state.text_queue.put(TTSEvent(type="text_start"))
                await self.state.text_queue.put(message.text)
                self.state.last_text_event_time = time.time()

                # Use create_task to track these tasks
                self.create_task(self.track_incoming_text())
                self.create_task(self.tts_model.text_stream_to_speech_to_queue(
                    text_queue=self.state.text_queue,
                    queue=self.state.audio_queue,
                    stream=True,
                ))
                self.create_task(self.process_audio_queue())
            else:
                await self.state.text_queue.put(message.text)
                self.state.last_text_event_time = time.time()
        elif message.type == "tts.text_done":
            await self.state.text_queue.put(TTSEvent(type="text_end"))
        elif message.type == "common.config_update":
            self.session_config = self.session_config.merge(message.config)



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


async def validate_message(raw_data: str) -> WSInterTTS:
    """Validate and parse incoming WebSocket messages"""
    try:
        json_data = json.loads(raw_data)
        event_type = json_data.get("type")
        if "common" in event_type:
            if "config_update" in event_type:
                return WSInterConfigUpdate(**json_data)
            elif "cancel" in event_type:
                return WSInterCancel(**json_data)
            elif "common.cleanup" in event_type:
                return WSInterCleanup(**json_data)
            else:
                raise Exception(f"Event of unrecognized type {event_type}")
        else:
            return WSInterTTS(**json_data)
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
            if parsed_message.type == "tts.add_text" and parsed_message.text:
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