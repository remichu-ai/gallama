from typing import Literal, Optional

from pydantic import BaseModel, validator
from .realtime_client_proto import SessionConfig
# this file contain websocket message schema for message sent between internal ws and tts, llm, stt


SpeechState = Literal["no_speech", "potential_speech", "is_speaking", "potential_speak_end", "speak_end"]


# common event
class WSInterConfigUpdate(BaseModel):
    type: Literal["common.config_update"]
    config: SessionConfig

class WSInterCancel(BaseModel):
    type: Literal["common.cancel"]


# STT
class WSInterSTT(BaseModel):
    type: Literal["stt.add_sound_chunk", "stt.sound_done", "stt.buffer_clear", "stt.one_time_transcribe"]
    sound: Optional[str] = None

class WSInterSTTResponse(BaseModel):
    type: Literal[
        "stt.add_transcription",
        "stt.transcription_complete",
        "stt.buffer_cleared",
        "stt.one_time_transcribe",
        "stt.vad_speech_start",
        "stt.vad_speech_end",
    ]
    transcription: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    vad_timestamp_ms: Optional[float] = None
    confidence: Optional[float] = None  # VAD confidence score


# TTS
class WSInterTTS(BaseModel):
    type: Literal["tts.add_text", "tts.text_done", "tts.interrupt"]
    text: Optional[str] = None

    @validator("text")
    def validate_text_for_add_type(cls, v, values):
        if values.get("type") == "add_text" and v is None:
            raise ValueError("text field cannot be None when type is add_text")
        return v


class TTSEvent(BaseModel):
    type: Literal["text_start", "text_end"]
