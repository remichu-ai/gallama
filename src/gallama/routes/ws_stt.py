from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile
import io
import librosa
from typing import Dict, Optional, Literal
from pydantic import BaseModel
from ..dependencies import get_model_manager
from ..data_classes.realtime_client_proto import SessionConfig
from ..data_classes.internal_ws import WSInterSTT, WSInterSTTResponse, WSInterConfigUpdate
from gallama.logger.logger import logger
import base64
import asyncio
import samplerate

router = APIRouter(prefix="", tags=["audio"])


class ConnectionData:
    def __init__(
        self,
        asr_processor,
        language: Optional[str],
        sample_rate: int,
        min_chunk_samples: int,
        streaming_mode: bool,
    ):
        self.asr_processor = asr_processor
        self.language = language
        self.raw_buffer = bytearray()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_chunk_samples = min_chunk_samples
        self.is_first = True
        self.sample_rate = sample_rate
        self.accumulated_duration = 0.0
        self.streaming_mode = streaming_mode
        self.complete_audio = bytearray() if not streaming_mode else None
        self.processing_complete = False
        self.transcription_enabled = True

    def reset_buffers(self):
        """Reset all buffers and state variables."""
        self.raw_buffer = bytearray()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.accumulated_duration = 0.0
        self.is_first = True
        if not self.streaming_mode:
            self.complete_audio = bytearray()
        self.processing_complete = False


class TranscriptionConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, ConnectionData] = {}
        self.MIN_CHUNK_SECONDS = 0.5

    async def connect(
        self,
        websocket: WebSocket,
        model: str,
        config: SessionConfig,
        language: str = None,
    ) -> bool:
        try:
            await websocket.accept()

            model_manager = get_model_manager()
            asr_processor = model_manager.stt_dict.get(model)
            if not asr_processor:
                try:
                    asr_processor = next(iter(model_manager.stt_dict.values()))
                except Exception as e:
                    logger.error(f"No STT model found in model_manager: {str(e)}")
                    asr_processor = None

            if asr_processor is None:
                await websocket.close(code=4000, reason="Model not found")
                return False

            asr_processor.reset_state()
            min_chunk_samples = int(self.MIN_CHUNK_SECONDS * config.input_sample_rate)

            # Create a new ConnectionData instance
            connection_data = ConnectionData(
                asr_processor=asr_processor,
                language=language,
                sample_rate=config.input_sample_rate,
                min_chunk_samples=min_chunk_samples,
                streaming_mode=config.streaming_transcription,
            )

            self.active_connections[websocket] = connection_data
            return True

        except Exception as e:
            logger.error(f"Error in connect: {str(e)}")
            await websocket.close(code=4000, reason="Connection initialization failed")
            return False

    async def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
        except Exception as e:
            logger.error(f"Error in disconnect: {str(e)}")

    async def process_one_time_transcription(self, websocket: WebSocket, audio_base64: str) -> bool:
        """Handle one-time transcription request"""
        connection = self.active_connections.get(websocket)
        if not connection:
            return False

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)

            # Process the audio directly without adding to the audio buffer
            processed_audio = self.process_raw_buffer(audio_bytes, connection.asr_processor, connection.sample_rate)
            if processed_audio is None:
                return False

            # Perform transcription
            transcription = await connection.asr_processor.transcribe_async(
                processed_audio,
                language=connection.language,
                include_segments=True
            )

            # Send response
            response = WSInterSTTResponse(
                type="stt.one_time_transcribe",
                transcription=transcription.text,
                start_time=0,
                end_time=len(processed_audio) / connection.asr_processor.SAMPLING_RATE
            )
            await websocket.send_json(response.dict())

            # Mark processing as complete
            connection.processing_complete = True
            return True

        except Exception as e:
            logger.error(f"Error in one-time transcription: {str(e)}")
            return False

    def process_raw_buffer(self, raw_buffer: bytearray, asr_processor, sample_rate: int) -> Optional[np.ndarray]:
        try:
            with soundfile.SoundFile(
                io.BytesIO(raw_buffer),
                channels=1,
                endian="LITTLE",
                samplerate=sample_rate,
                subtype="PCM_16",
                format="RAW"
            ) as sf:
                audio, curr_sr = librosa.load(sf, sr=asr_processor.SAMPLING_RATE, dtype=np.float32)
                return audio
        except Exception as e:
            logger.error(f"Error processing raw audio: {str(e)}")
            return None

    async def process_audio_chunk(self, websocket: WebSocket, audio_chunk: bytes, is_final: bool = False) -> bool:
        connection = self.active_connections.get(websocket)
        if not connection:
            return False

        # Add check for transcription_enabled
        if not connection.transcription_enabled:
            return True  # Successfully ignored chunk due to disabled transcription

        try:
            if connection.streaming_mode:
                # Streaming mode - process chunks as they come
                success, is_final_from_vad = await self._process_streaming_chunk(connection, websocket, audio_chunk, is_final)
                if is_final or is_final_from_vad:
                    connection.processing_complete = True

                    # send completion event to client
                    complete_response = WSInterSTTResponse(type="stt.transcription_complete")
                    await websocket.send_json(complete_response.model_dump())
                    logger.info("Send completion event for VAD based transcription")
                return success
            else:
                # One-shot mode - accumulate all audio
                connection.complete_audio.extend(audio_chunk)
                if is_final:
                    logger.info("Processing complete audio in one-shot mode")
                    success = await self._process_complete_audio(connection, websocket)
                    if success:
                        logger.info("One-shot audio processing completed successfully")
                    else:
                        logger.warning("One-shot audio processing failed")
                    connection.processing_complete = True
                    return success
                return True

        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            await websocket.close(code=4000, reason="Processing error")
            return False

    async def _process_streaming_chunk(self, connection: ConnectionData, websocket: WebSocket, audio_chunk: bytes, is_final: bool):
        logger.info(f"Processing {'final' if is_final else 'intermediate'} streaming chunk")
        is_final_from_vad = False

        # Calculate minimum samples needed for good resampling
        # Using a minimum of 1024 samples for resampling window
        RESAMPLING_WINDOW = max(1024, int(0.05 * connection.sample_rate))  # At least 50ms of audio

        # Add new audio chunk to raw buffer
        if audio_chunk:
            connection.raw_buffer.extend(audio_chunk)

        # Only process if we have enough samples or if this is the final chunk
        if len(connection.raw_buffer) >= (RESAMPLING_WINDOW * 2) or (is_final and len(connection.raw_buffer) > 0):
            # Process the audio with proper buffering
            processed_audio = self.process_raw_buffer(connection.raw_buffer, connection.asr_processor, connection.sample_rate)
            if processed_audio is None:
                return False, False

            # Add processed audio to the buffer
            connection.audio_buffer = np.concatenate([connection.audio_buffer, processed_audio])

            # Clear the raw buffer since we've processed it
            connection.raw_buffer = bytearray()

            # Process accumulated audio if we have enough samples or if this is final
            if (len(connection.audio_buffer) >= connection.min_chunk_samples) or (is_final and len(connection.audio_buffer) > 0):
                connection.asr_processor.add_audio_chunk(connection.audio_buffer)
                start_time, end_time, transcription, vad_events = connection.asr_processor.process_audio(is_final=is_final)

                # Handle VAD events
                for event in vad_events:
                    if event['type'] == 'start':
                        response = WSInterSTTResponse(
                            type="stt.vad_speech_start",
                            vad_timestamp_ms=event['timestamp_ms'],
                            confidence=event['confidence']
                        )
                        await websocket.send_json(response.dict())
                        await asyncio.sleep(0.1)

                    if event['type'] == 'end':
                        response = WSInterSTTResponse(
                            type="stt.vad_speech_end",
                            vad_timestamp_ms=event['timestamp_ms'],
                            confidence=event['confidence']
                        )
                        await websocket.send_json(response.dict())
                        is_final_from_vad = True

                # Send transcription if available
                if transcription:
                    response = WSInterSTTResponse(
                        type="stt.add_transcription",
                        transcription=transcription,
                        start_time=start_time,
                        end_time=end_time
                    )
                    await websocket.send_json(response.dict())

                # Reset buffers based on whether this is final
                if not is_final:
                    connection.audio_buffer = np.array([], dtype=np.float32)
                    connection.is_first = False
                else:
                    connection.audio_buffer = np.array([], dtype=np.float32)
                    connection.is_first = True

        return True, is_final_from_vad

    async def _process_complete_audio(self, connection: ConnectionData, websocket: WebSocket):
        """Handle one-shot audio processing"""
        processed_audio = self.process_raw_buffer(connection.complete_audio, connection.asr_processor, connection.sample_rate)
        if processed_audio is None:
            return False

        try:
            transcription = await connection.asr_processor.transcribe_async(
                processed_audio,
                language=connection.language,
                include_segments=True
            )

            response = WSInterSTTResponse(
                type="stt.add_transcription",
                transcription=transcription.text,
                start_time=0,
                end_time=len(processed_audio) / connection.asr_processor.SAMPLING_RATE
            )
            await websocket.send_json(response.dict())
            return True

        except Exception as e:
            logger.error(f"Error in one-shot transcription: {str(e)}")
            return False

    async def is_processing_complete(self, websocket: WebSocket) -> bool:
        """Check if audio processing is complete for a given websocket connection"""
        connection = self.active_connections.get(websocket)
        if not connection:
            return True  # If connection doesn't exist, consider it complete
        return connection.processing_complete


@router.websocket("/speech-to-text")
async def websocket_endpoint(
    websocket: WebSocket,
    model: str = None,
    language: str = None,
    config: SessionConfig = None
):
    if config is None:
        config = SessionConfig()

    try:
        success = await manager.connect(websocket, model, config, language)
        if not success:
            return

        while True:
            data = await websocket.receive_json()
            if "stt." in data["type"]:
                message = WSInterSTT.model_validate(data)
            elif "common." in data["type"]:
                message = WSInterConfigUpdate.model_validate(data)

            if message.type == "stt.add_sound_chunk" and message.sound:
                audio_bytes = base64.b64decode(message.sound)
                await manager.process_audio_chunk(websocket, audio_bytes, is_final=False)

            elif message.type == "stt.sound_done":
                logger.info("STT: received sound_done message.")
                connection = manager.active_connections.get(websocket)

                try:
                    if connection:
                        # Process the final chunk
                        await manager.process_audio_chunk(websocket, b"", is_final=True)

                        # Wait until processing is complete before sending transcription_complete
                        while not await manager.is_processing_complete(websocket):
                            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

                except Exception as e:
                    logger.error(f"Error during sound_done processing: {str(e)}")
                    raise  # Re-raise the exception to trigger proper cleanup

            elif message.type == "stt.buffer_clear" or message.type == "common.cancel":
                logger.info("STT: received clear_buffer message.")

                connection = manager.active_connections.get(websocket)
                if connection:
                    # Reset the ASR processor state
                    connection.asr_processor.reset_state()
                    # Reset all buffers
                    connection.reset_buffers()

                    # Send confirmation back to client
                    clear_response = WSInterSTTResponse(type="stt.buffer_cleared")
                    await websocket.send_json(clear_response.model_dump())
            elif message.type == "stt.one_time_transcribe":
                if not message.sound:
                    logger.error("Received one_time_transcribe without audio data")
                    continue

                success = await manager.process_one_time_transcription(websocket, message.sound)
                if not success:
                    logger.error("Failed to process one-time transcription")
            elif message.type == "common.config_update":
                connection = manager.active_connections.get(websocket)
                logger.info(f"Received config update in STT: {message.config}")
                if connection:
                    # Reset the ASR processor state
                    connection.asr_processor.update_vad_config(message.config.turn_detection)
                    connection.asr_processor.reset_state()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
    finally:
        await manager.disconnect(websocket)


manager = TranscriptionConnectionManager()

