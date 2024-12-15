from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import soundfile
import io
import librosa
from typing import Dict, Optional
from ..dependencies import get_model_manager
from gallama.logger.logger import logger

router = APIRouter(prefix="", tags=["audio"])

MIN_CHUNK_SECONDS = 0.5  # Minimum chunk size in seconds


class TranscriptionConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict] = {}

    async def connect(
            self,
            websocket: WebSocket,
            model: str,
            language: str = None
    ) -> bool:
        try:
            await websocket.accept()

            model_manager = get_model_manager()
            asr_processor = model_manager.stt_dict.get(model)

            if asr_processor is None:
                await websocket.close(code=4000, reason="Model not found")
                return False

            # Initialize ASR state and connection data
            asr_processor.reset_state()
            min_chunk_samples = int(MIN_CHUNK_SECONDS * asr_processor.SAMPLING_RATE)

            self.active_connections[websocket] = {
                "asr_processor": asr_processor,
                "language": language,
                "raw_buffer": bytearray(),  # Store raw bytes instead of processed audio
                "min_chunk_samples": min_chunk_samples,
                "is_first": True
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

    def process_raw_buffer(self, raw_buffer: bytearray, asr_processor) -> Optional[np.ndarray]:
        """Convert accumulated raw bytes to numpy array with correct sampling rate"""
        try:
            # Create a SoundFile object from the accumulated raw bytes
            with soundfile.SoundFile(
                    io.BytesIO(raw_buffer),
                    channels=1,
                    endian="LITTLE",
                    samplerate=asr_processor.SAMPLING_RATE,
                    subtype="PCM_16",
                    format="RAW"
            ) as sf:
                audio, _ = librosa.load(sf, sr=asr_processor.SAMPLING_RATE, dtype=np.float32)
                return audio
        except Exception as e:
            logger.error(f"Error processing raw audio: {str(e)}")
            return None

    async def process_audio_chunk(self, websocket: WebSocket, audio_chunk: bytes) -> bool:
        connection = self.active_connections.get(websocket)
        if not connection:
            return False

        asr_processor = connection["asr_processor"]
        raw_buffer = connection["raw_buffer"]
        min_chunk_samples = connection["min_chunk_samples"]
        is_first = connection["is_first"]

        try:
            # Accumulate raw bytes
            raw_buffer.extend(audio_chunk)

            # Calculate number of samples based on 16-bit PCM
            num_samples = len(raw_buffer) // 2  # 2 bytes per sample for PCM_16

            # Process if we have enough samples or not first chunk
            if num_samples >= min_chunk_samples or not is_first:
                # Convert accumulated raw bytes to audio
                audio_data = self.process_raw_buffer(raw_buffer, asr_processor)
                if audio_data is None:
                    return False

                # Add to ASR processor
                asr_processor.add_audio_chunk(audio_data)

                # Process the audio and get transcription
                start_time, end_time, transcription = asr_processor.process_audio()

                if transcription:
                    response = {
                        "text": transcription,
                        "start": start_time,
                        "end": end_time,
                        "is_final": False
                    }
                    await websocket.send_json(response)

                # Clear buffer after processing
                connection["raw_buffer"] = bytearray()
                connection["is_first"] = False

                return True

            connection["raw_buffer"] = raw_buffer  # Store updated buffer
            return True

        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            await websocket.close(code=4000, reason="Processing error")
            return False


manager = TranscriptionConnectionManager()


@router.websocket("/stream-transcription")
async def websocket_endpoint(
        websocket: WebSocket,
        model: str,
        language: str = None
):
    """
    WebSocket endpoint for streaming audio transcription.
    Expects 16-bit PCM audio at 16kHz
    """
    try:
        success = await manager.connect(websocket, model, language)
        if not success:
            return

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.receive":
                if "bytes" in message:
                    success = await manager.process_audio_chunk(websocket, message["bytes"])
                    if not success:
                        break

                elif message.get("text") == "EOS":
                    # Process any remaining audio in buffer
                    connection = manager.active_connections.get(websocket)
                    if connection and len(connection["raw_buffer"]) > 0:
                        audio_data = manager.process_raw_buffer(
                            connection["raw_buffer"],
                            connection["asr_processor"]
                        )
                        if audio_data is not None:
                            asr_processor = connection["asr_processor"]
                            asr_processor.add_audio_chunk(audio_data)
                            start_time, end_time, transcription = asr_processor.process_audio()
                            if transcription:
                                await websocket.send_json({
                                    "text": transcription,
                                    "start": start_time,
                                    "end": end_time,
                                    "is_final": True
                                })
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=4000, reason="Internal server error")
        except:
            pass
    finally:
        await manager.disconnect(websocket)