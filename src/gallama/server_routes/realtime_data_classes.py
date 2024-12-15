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


# class AudioBuffer:
#     def __init__(self):
#         self.buffer = bytearray()
#         self.max_size = 15 * 1024 * 1024  # 15 MiB
#
#     def append(self, audio_data: bytes) -> bool:
#         if len(self.buffer) + len(audio_data) > self.max_size:
#             return False
#         self.buffer.extend(audio_data)
#         return True
#
#     def clear(self):
#         self.buffer = bytearray()
#
#     def get_buffer(self) -> bytes:
#         return bytes(self.buffer)
#
#     def is_empty(self) -> bool:
#         return len(self.buffer) == 0

# class ConversationManager:
#     def __init__(self):
#         self.conversations: Dict[str, List[ConversationItem]] = {}
#         self.active_responses: Dict[str, bool] = {}
#
#     def get_or_create_conversation(self, session_id: str) -> List[ConversationItem]:
#         if session_id not in self.conversations:
#             self.conversations[session_id] = []
#         return self.conversations[session_id]
#
#     def add_item(self, session_id: str, item: ConversationItem, previous_item_id: Optional[str] = None) -> tuple[
#         bool, str]:
#         conversation = self.get_or_create_conversation(session_id)
#
#         if not item.id:
#             item.id = f"msg_{len(conversation)}_{int(datetime.now().timestamp())}"
#
#         if previous_item_id:
#             for i, existing_item in enumerate(conversation):
#                 if existing_item.id == previous_item_id:
#                     conversation.insert(i + 1, item)
#                     return True, item.id
#             return False, "Previous item not found"
#
#         conversation.append(item)
#         return True, item.id
#
#     def delete_item(self, session_id: str, item_id: str) -> bool:
#         conversation = self.get_or_create_conversation(session_id)
#         for i, item in enumerate(conversation):
#             if item.id == item_id:
#                 conversation.pop(i)
#                 return True
#         return False
#
#     def truncate_item(self, session_id: str, item_id: str, content_index: int, audio_end_ms: int) -> bool:
#         conversation = self.get_or_create_conversation(session_id)
#         for item in conversation:
#             if item.id == item_id and item.role == Role.ASSISTANT:
#                 # In a real implementation, you would truncate the audio here
#                 return True
#         return False
#


# class SessionManager:
#     def __init__(self):
#         self.sessions: Dict[str, SessionConfig] = {}
#         self.conversation_manager = ConversationManager()
#         self.audio_buffers: Dict[str, AudioBuffer] = {}
#
#     async def update_session(self, websocket: WebSocket, session_id: str, update: SessionConfig) -> SessionConfig:
#         current_session = self.sessions.get(session_id, SessionConfig(modalities=["text"]))
#
#         # Update only the fields that are present in the update
#         updated_data = current_session.dict()
#         update_data = update.dict(exclude_unset=True)
#         updated_data.update(update_data)
#
#         updated_session = SessionConfig(**updated_data)
#         self.sessions[session_id] = updated_session
#
#         # Send session.updated event
#         response = {
#             "type": "session.updated",
#             "session": updated_session.dict()
#         }
#         await websocket.send_text(json.dumps(response))
#         return updated_session
#
#     async def create_response(self, websocket: WebSocket, session_id: str, config: SessionConfig):
#         # Set response as active
#         self.conversation_manager.active_responses[session_id] = True
#
#         try:
#             # Send response.created event
#             await websocket.send_json({
#                 "type": "response.created"
#             })
#
#             # Here you would implement your LLM logic
#             # For now, we'll just send a dummy response
#
#             # Send conversation.item.created for the response
#             item_id = f"msg_response_{int(datetime.now().timestamp())}"
#             await websocket.send_json({
#                 "type": "conversation.item.created",
#                 "item": {
#                     "id": item_id,
#                     "type": "message",
#                     "role": "assistant",
#                     "content": [{"type": "text", "text": "This is a placeholder response."}]
#                 }
#             })
#
#             # Send response.done event
#             await websocket.send_json({
#                 "type": "response.done"
#             })
#
#         finally:
#             # Clear active response
#             self.conversation_manager.active_responses[session_id] = False
#
#     async def cancel_response(self, websocket: WebSocket, session_id: str) -> bool:
#         if not self.conversation_manager.active_responses.get(session_id):
#             return False
#
#         self.conversation_manager.active_responses[session_id] = False
#         await websocket.send_json({
#             "type": "response.cancelled"
#         })
#         return True


