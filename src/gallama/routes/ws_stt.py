from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile
import io
import librosa
from typing import Dict, Optional, Literal
from pydantic import BaseModel
from ..dependencies import get_model_manager
from ..data_classes.internal_ws import WSInterSTT, WSInterSTTResponse
from gallama.logger.logger import logger
import base64

router = APIRouter(prefix="", tags=["audio"])



class TranscriptionConnectionManager:

    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict] = {}
        self.MIN_CHUNK_SECONDS = 0.5

    async def connect(
        self,
        websocket: WebSocket,
        model: str,
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
                "audio_buffer": np.array([], dtype=np.float32),  # New buffer for processed audio
                "min_chunk_samples": min_chunk_samples,
                "is_first": True,
                "sample_rate": sample_rate,
                "accumulated_duration": 0.0  # Track accumulated audio duration
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

        asr_processor = connection["asr_processor"]
        raw_buffer = connection["raw_buffer"]
        audio_buffer = connection["audio_buffer"]
        min_chunk_samples = connection["min_chunk_samples"]
        sample_rate = connection["sample_rate"]

        try:
            # Process incoming audio chunk
            raw_buffer.extend(audio_chunk)
            processed_audio = self.process_raw_buffer(raw_buffer, asr_processor, sample_rate)
            if processed_audio is None:
                return False

            # Append to audio buffer
            audio_buffer = np.concatenate([audio_buffer, processed_audio])
            connection["audio_buffer"] = audio_buffer
            connection["accumulated_duration"] = len(audio_buffer) / asr_processor.SAMPLING_RATE

            # Only process if we have enough audio or it's the final chunk
            if len(audio_buffer) >= min_chunk_samples or is_final:
                asr_processor.add_audio_chunk(audio_buffer)
                start_time, end_time, transcription = asr_processor.process_audio(is_final=is_final)

                if transcription:
                    response = WSInterSTTResponse(
                        type="stt.add_transcription",
                        transcription=transcription,
                        start_time=start_time,
                        end_time=end_time
                    )
                    await websocket.send_json(response.dict())

                # Reset buffers
                connection["raw_buffer"] = bytearray()
                connection["audio_buffer"] = np.array([], dtype=np.float32)
                connection["accumulated_duration"] = 0.0
                connection["is_first"] = False

        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            await websocket.close(code=4000, reason="Processing error")
            return False


manager = TranscriptionConnectionManager()


@router.websocket("/speech-to-text")
async def websocket_endpoint(
        websocket: WebSocket,
        model: str = None,
        language: str = None,
        sample_rate: int = 16000
):
    try:
        success = await manager.connect(websocket, model, language, sample_rate)
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
                        # Process any remaining audio
                        await manager.process_audio_chunk(websocket, b"", is_final=True)

                    # Always send completion message, regardless of connection or processing status
                    complete_response = WSInterSTTResponse(type="stt.transcription_complete")
                    await websocket.send_json(complete_response.dict())

                except Exception as e:
                    logger.error(f"Error during sound_done processing: {str(e)}")
                    # Still try to send completion message even if processing failed
                    complete_response = WSInterSTTResponse(type="stt.transcription_complete")
                    await websocket.send_json(complete_response.dict())

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        await manager.disconnect(websocket)