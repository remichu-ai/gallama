from typing import Literal, Optional

from pydantic import BaseModel, validator
from .realtime_data_classes import SessionConfig
# this file contain websocket message schema for message sent between internal ws and tts, llm, stt


# common event
class WSInterConfigUpdate(BaseModel):
    type: Literal["common.config_update"]
    config: SessionConfig


# STT
class WSInterSTT(BaseModel):
    type: Literal["stt.add_sound_chunk", "stt.sound_done", "stt.clear_buffer"]
    sound: Optional[str] = None

class WSInterSTTResponse(BaseModel):
    type: Literal["stt.add_transcription", "stt.transcription_complete", "stt.clear_buffer_done", "stt.transcription_complete"]
    transcription: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


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
