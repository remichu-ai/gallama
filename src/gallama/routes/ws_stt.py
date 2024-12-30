from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile
import io
import librosa
from typing import Dict, Optional, Literal
from pydantic import BaseModel
from ..dependencies import get_model_manager
from ..data_classes.realtime_client_proto import SessionConfig
from ..data_classes.internal_ws import WSInterSTT, WSInterSTTResponse
from gallama.logger.logger import logger
import base64
import asyncio

router = APIRouter(prefix="", tags=["audio"])


class TranscriptionConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict] = {}
        self.MIN_CHUNK_SECONDS = 0.5

    async def connect(
        self,
        websocket: WebSocket,
        model: str,
        config: SessionConfig,
        language: str = None,
        sample_rate: int = 16000
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
            min_chunk_samples = int(self.MIN_CHUNK_SECONDS * sample_rate)

            self.active_connections[websocket] = {
                "asr_processor": asr_processor,
                "language": language,
                "raw_buffer": bytearray(),
                "audio_buffer": np.array([], dtype=np.float32),
                "min_chunk_samples": min_chunk_samples,
                "is_first": True,
                "sample_rate": sample_rate,
                "accumulated_duration": 0.0,
                "streaming_mode": config.streaming_transcription,
                "complete_audio": bytearray() if not config.streaming_transcription else None,
                "processing_complete": False,
                "transcription_enabled": True  # Add this line
            }
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

        asr_processor = connection["asr_processor"]
        sample_rate = connection["sample_rate"]

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)

            # Process the audio
            processed_audio = self.process_raw_buffer(audio_bytes, asr_processor, sample_rate)
            if processed_audio is None:
                return False

            # Perform transcription
            transcription = await asr_processor.transcribe_async(
                processed_audio,
                language=connection["language"],
                include_segments=True
            )

            # Send response
            response = WSInterSTTResponse(
                type="stt.one_time_transcribe",
                transcription=transcription.text,
                start_time=0,
                end_time=len(processed_audio) / asr_processor.SAMPLING_RATE
            )
            await websocket.send_json(response.dict())

            # Mark processing as complete
            connection["processing_complete"] = True
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
        if not connection.get("transcription_enabled", True):
            return True  # Successfully ignored chunk due to disabled transcription

        asr_processor = connection["asr_processor"]
        streaming_mode = connection["streaming_mode"]

        try:
            if streaming_mode:
                # Streaming mode - process chunks as they come
                success = await self._process_streaming_chunk(connection, websocket, audio_chunk, is_final)
                if is_final:
                    connection["processing_complete"] = True
                return success
            else:
                # One-shot mode - accumulate all audio
                connection["complete_audio"].extend(audio_chunk)
                if is_final:
                    logger.info("Processing complete audio in one-shot mode")
                    success = await self._process_complete_audio(connection, websocket)
                    if success:
                        logger.info("One-shot audio processing completed successfully")
                    else:
                        logger.warning("One-shot audio processing failed")
                    connection["processing_complete"] = True
                    return success
                return True

        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            await websocket.close(code=4000, reason="Processing error")
            return False

    async def _process_streaming_chunk(self, connection, websocket, audio_chunk, is_final):
        """Handle streaming mode audio processing"""
        logger.info(f"Processing {'final' if is_final else 'intermediate'} streaming chunk")
        raw_buffer = connection["raw_buffer"]
        audio_buffer = connection["audio_buffer"]
        min_chunk_samples = connection["min_chunk_samples"]
        sample_rate = connection["sample_rate"]
        asr_processor = connection["asr_processor"]

        raw_buffer.extend(audio_chunk)
        processed_audio = self.process_raw_buffer(raw_buffer, asr_processor, sample_rate)
        if processed_audio is None:
            return False

        audio_buffer = np.concatenate([audio_buffer, processed_audio])
        connection["audio_buffer"] = audio_buffer
        connection["accumulated_duration"] = len(audio_buffer) / asr_processor.SAMPLING_RATE

        if len(audio_buffer) >= min_chunk_samples or is_final:
            asr_processor.add_audio_chunk(audio_buffer)
            start_time, end_time, transcription = asr_processor.process_audio(is_final=is_final)

            if transcription:
                logger.info(f"Got transcription for chunk: {transcription[:50]}... (start: {start_time}, end: {end_time})")
                response = WSInterSTTResponse(
                    type="stt.add_transcription",
                    transcription=transcription,
                    start_time=start_time,
                    end_time=end_time
                )
                await websocket.send_json(response.dict())
                logger.debug("Sent transcription response")

            connection["raw_buffer"] = bytearray()
            connection["audio_buffer"] = np.array([], dtype=np.float32)
            connection["accumulated_duration"] = 0.0
            connection["is_first"] = False

        return True

    async def _process_complete_audio(self, connection, websocket):
        """Handle one-shot audio processing"""
        asr_processor = connection["asr_processor"]
        complete_audio = connection["complete_audio"]
        sample_rate = connection["sample_rate"]

        processed_audio = self.process_raw_buffer(complete_audio, asr_processor, sample_rate)
        if processed_audio is None:
            return False

        try:
            transcription = await asr_processor.transcribe_async(
                processed_audio,
                language=connection["language"],
                include_segments=True
            )

            response = WSInterSTTResponse(
                type="stt.add_transcription",
                transcription=transcription.text,
                start_time=0,
                end_time=len(processed_audio) / asr_processor.SAMPLING_RATE
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
        return connection.get("processing_complete", False)


@router.websocket("/speech-to-text")
async def websocket_endpoint(
        websocket: WebSocket,
        model: str = None,
        language: str = None,
        sample_rate: int = 16000,
        config: SessionConfig = None
):
    if config is None:
        config = SessionConfig()

    try:
        success = await manager.connect(websocket, model, config, language, sample_rate)
        if not success:
            return

        while True:
            data = await websocket.receive_json()
            message = WSInterSTT.parse_obj(data)

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

                        complete_response = WSInterSTTResponse(type="stt.transcription_complete")
                        await websocket.send_json(complete_response.dict())

                except Exception as e:
                    logger.error(f"Error during sound_done processing: {str(e)}")
                    raise  # Re-raise the exception to trigger proper cleanup


            elif message.type == "stt.buffer_clear" or message.type=="common.cancel":
                logger.info("STT: received clear_buffer message.")

                connection = manager.active_connections.get(websocket)
                if connection:
                    # Reset the ASR processor state
                    connection["asr_processor"].reset_state()
                    # Reset all buffers
                    connection["raw_buffer"] = bytearray()
                    connection["audio_buffer"] = np.array([], dtype=np.float32)
                    connection["accumulated_duration"] = 0.0
                    connection["is_first"] = True

                    if not connection["streaming_mode"]:
                        connection["complete_audio"] = bytearray()
                    connection["processing_complete"] = False

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



    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
    finally:
        await manager.disconnect(websocket)


manager = TranscriptionConnectionManager()
