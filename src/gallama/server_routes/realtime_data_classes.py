from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum
import json
import asyncio
import base64
from datetime import datetime


# Enums and Pydantic models
class AudioFormat(str, Enum):
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"


class Voice(str, Enum):
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    SAGE = "sage"
    VERSE = "verse"
    ALLOY = "alloy"
    ECHO = "echo"
    SHIMMER = "shimmer"


class ToolChoice(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class AudioTranscriptionConfig(BaseModel):
    model: Literal["whisper-1"]


class TurnDetectionConfig(BaseModel):
    type: Literal["server_vad"]
    threshold: float = Field(ge=0.0, le=1.0)
    prefix_padding_ms: int = Field(ge=0)
    silence_duration_ms: int = Field(ge=0)


class ToolParameter(BaseModel):
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]


class Tool(BaseModel):
    type: Literal["function"]
    name: str
    description: str
    parameters: ToolParameter

class AudioBufferAppend(BaseModel):
    event_id: Optional[str] = None
    type: Literal["input_audio_buffer.append"]
    audio: str  # Base64 encoded audio data

class AudioBufferCommit(BaseModel):
    event_id: Optional[str] = None
    type: Literal["input_audio_buffer.commit"]

class AudioBufferClear(BaseModel):
    event_id: Optional[str] = None
    type: Literal["input_audio_buffer.clear"]


class SessionConfig(BaseModel):
    modalities: List[Literal["text", "audio"]]
    instructions: Optional[str] = ""    # system prompt
    voice: Optional[Voice] = None
    input_audio_format: Optional[AudioFormat] = None
    output_audio_format: Optional[AudioFormat] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    turn_detection: Optional[TurnDetectionConfig] = None
    tools: Optional[List[Tool]] = []
    tool_choice: Optional[Union[ToolChoice, str]] = "auto"
    temperature: Optional[float] = Field(0.8, ge=0.6, le=1.2)
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = "inf"

    # extra
    model: Optional[str] = None

    @validator('max_response_output_tokens')
    def validate_max_tokens(cls, v):
        if isinstance(v, int) and (v < 1 or v > 4096):
            raise ValueError("max_response_output_tokens must be between 1 and 4096 or 'inf'")
        return v

class ContentType(str, Enum):
    INPUT_TEXT = "input_text"
    INPUT_AUDIO = "input_audio"
    TEXT = "text"

class ItemType(str, Enum):
    MESSAGE = "message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ItemStatus(str, Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"

class MessageContent(BaseModel):
    type: ContentType
    text: Optional[str] = None
    audio: Optional[str] = None

class ConversationItem(BaseModel):
    id: Optional[str] = None
    type: ItemType
    object: str = "realtime.item"
    status: ItemStatus = ItemStatus.COMPLETED
    role: Optional[Role] = None
    content: Optional[List[MessageContent]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None

class ConversationItemCreate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.create"]
    previous_item_id: Optional[str] = None
    item: ConversationItem

class ConversationItemTruncate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.truncate"]
    item_id: str
    content_index: int
    audio_end_ms: int

class ConversationItemDelete(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.delete"]
    item_id: str

class ResponseCreate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["response.create"]
    response: SessionConfig

class ResponseCancel(BaseModel):
    event_id: Optional[str] = None
    type: Literal["response.cancel"]

class SessionCreated(BaseModel):
    event_id: Optional[str] = None
    type: Literal["session.created"]
    session: SessionConfig


class SessionUpdate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["session.update"]
    session: SessionConfig


