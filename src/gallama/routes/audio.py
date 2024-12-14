from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import soundfile as sf
from gallama.data_classes import (
    TranscriptionResponse,
    TTSRequest
)

from typing import List, Literal, Optional
import asyncio
from ..dependencies import get_model_manager

# https://platform.openai.com/docs/api-reference/audio

router = APIRouter(prefix="/v1/audio", tags=["audio"])

@router.post("/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe."),
    model: str = Form(..., description="ID of the model to use."),
    language: Optional[str] = Form(None, description="The language of the input audio in ISO-639-1 format."),
    prompt: Optional[str] = Form(None, description="An optional text to guide the model's style or continue a previous segment."),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Form("json", description="The format of the output."),
    temperature: Optional[float] = Form(0.0, description="The sampling temperature, between 0 and 1."),
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Form(["segment"], description="The timestamp granularities for transcription.")
):
    """
    Transcribe an audio file into the input language.
    """
    model_manager = get_model_manager()
    stt = model_manager.stt_dict.get(model)

    if stt is None:
        raise HTTPException(status_code=400, detail="Model not found")

    if response_format not in  ["json", "verbose_json"]:
        raise HTTPException(status_code=400, detail="Invalid response format, currently only json format supported")

    include_segments = False
    if response_format == "verbose_json" and "segment" in timestamp_granularities:
        include_segments = True

    transcribed_object = await stt.transcribe_async(
        audio=file.file,
        init_prompt=prompt,
        temperature=temperature,
        language=language,
        include_segments=include_segments,
    )

    return transcribed_object

@router.post("/speech")
async def create_speech(request: TTSRequest):
    # Validate the model, voice, and input length
    # if request.model not in SUPPORTED_MODELS:
    #     raise HTTPException(status_code=400, detail="Unsupported model")
    # if request.voice not in SUPPORTED_VOICES:
    #     raise HTTPException(status_code=400, detail="Unsupported voice")
    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds the maximum length of 4096 characters")

    global tts_dict
    model_manager = get_model_manager()
    tts = model_manager.tts_dict[request.model]

    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

    if request.speed < 0.25 or request.speed > 4.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")

    # Get the current event loop at the start
    loop = asyncio.get_running_loop()

    # Generate audio asynchronously
    #voice_desc = get_voice_description(request.voice)
    sampling_rate, audio_data = await tts.text_to_speech(
        text = request.input,
        stream = False,
        batching = True,
        batch_size = 3
    )

    # Convert to desired format using a thread pool
    buffer = io.BytesIO()
    if request.response_format not in ["mp3", "wav", "flac"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.response_format}")

    # Write audio data to buffer in a thread pool
    await loop.run_in_executor(
        None,
        lambda: sf.write(buffer, audio_data, sampling_rate, format=request.response_format)
    )
    buffer.seek(0)

    # Set the appropriate media type
    media_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac"
    }

    return StreamingResponse(
        buffer,
        media_type=media_types[request.response_format],
        headers={
            "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
        }
    )