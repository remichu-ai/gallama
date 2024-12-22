from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional
from fastapi import Query


# class for audio api call
class TranscriptionResponse(BaseModel):
    task: Optional[str] = Field(default="transcribe", description="The task performed.")
    language: Optional[str] = Field(default=None, description="The detected or specified language of the audio.")
    duration: Optional[float] = Field(default=None, description="The duration of the audio in seconds.")
    text: str = Field(description="The transcription text.")
    words: Optional[List[dict]] = Field(default=None, description="Word-level timestamps, if requested.")
    segments: Optional[List[dict]] = Field(default=None, description="Segment-level timestamps, if requested.")


# class for coding
class TimeStampedWord(BaseModel):
    pass


# Define supported models, voices, and formats
SUPPORTED_MODELS = {"tts-1", "tts-1-hd"}
SUPPORTED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
DEFAULT_FORMAT = "wav"

# Pydantic model for request validation for text to speech
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    response_format: str = Query(DEFAULT_FORMAT, enum=list(SUPPORTED_FORMATS))
    speed: float = Query(1.0, ge=0.25, le=4.0)



class TTSEvent(BaseModel):
    type: Literal["text_start", "text_end"]


class WSMessageTTS(BaseModel):
    type: Literal["add_text", "text_done", "interrupt"]
    text: Optional[str] = None

    @validator("text")
    def validate_text_for_add_type(cls, v, values):
        if values.get("type") == "add_text" and v is None:
            raise ValueError("text field cannot be None when type is add_text")
        return v
