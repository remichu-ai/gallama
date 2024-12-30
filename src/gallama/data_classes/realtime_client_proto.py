from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum


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
    model: Literal["whisper-1"] = "whisper-1"


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


class TurnDetectionConfig(BaseModel):
    type: Literal["server_vad"] = "server_vad"
    threshold: Optional[float] = Field(ge=0.0, le=1.0,default=0.5)
    prefix_padding_ms: Optional[int] = Field(ge=0, default=700)
    silence_duration_ms: Optional[int] = Field(ge=0, default=500)
    create_response: bool = True
    enable_preprocessing: Optional[bool] = True

class SessionConfig(BaseModel):
    modalities: List[Literal["text", "audio"]] = Field(default_factory=lambda: ["text", "audio"])
    instructions: Optional[str] = ""    # system prompt
    voice: Optional[str] = None
    input_audio_format: Optional[AudioFormat] = None
    output_audio_format: Optional[AudioFormat] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    turn_detection: Optional[TurnDetectionConfig] = Field(default_factory=TurnDetectionConfig)
    tools: Optional[List[Tool]] = Field(default_factory=list)
    tool_choice: Optional[Union[ToolChoice, str]] = "auto"
    temperature: Optional[float] = Field(0.4, ge=0.1, le=1.2)
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = "inf"

    # extra
    model: Optional[str] = None

    # extra for gallama backend
    streaming_transcription: bool = True
    user_interrupt_token: Optional[str] = Field(description= "Custom word to insert everytime user interrupt the assistant",default=" <user_interrupt>")

    class Config:
        extra = "allow"  # Allow extra fields

    @validator('max_response_output_tokens')
    def validate_max_tokens(cls, v):
        if isinstance(v, int) and (v < 1 or v > 4096):
            raise ValueError("max_response_output_tokens must be between 1 and 4096 or 'inf'")
        return v

    def merge(self, other: Union[dict, "SessionConfig"]) -> "SessionConfig":
        """
        Merge this SessionConfig with another SessionConfig or dictionary.
        The values from 'other' will overwrite existing values if present.

        Args:
            other: Either a dictionary or another SessionConfig instance containing
                  configuration values to merge.

        Returns:
            A new SessionConfig instance with merged values.
        """
        # Convert self to dict
        current_config = self.dict()

        # If other is a SessionConfig instance, convert it to dict
        if isinstance(other, SessionConfig):
            other_config = other.dict()
        else:
            other_config = other

        # Merge the dictionaries, with other_config taking precedence
        merged_config = {
            **current_config,
            **{k: v for k, v in other_config.items() if v is not None}
        }

        # Create and return a new SessionConfig instance
        return SessionConfig(**merged_config)


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


class ConversationItemBase(BaseModel):
    id: Optional[str] = None
    object: Literal["realtime.item"] = "realtime.item"
    status: Optional[Literal["completed", "incomplete"]] = None


class ConversationItemMessage(ConversationItemBase):
    type: Literal["message"] = "message"
    role: Role
    content: List[MessageContent]

class ConversationItemFunctionCall(ConversationItemBase):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str

class ConversationItemFunctionCallOutput(ConversationItemBase):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

# Define the union
ConversationItem = Union[
    ConversationItemMessage,
    ConversationItemFunctionCall,
    ConversationItemFunctionCallOutput
]


class ConversationItemCreate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: Optional[str] = None
    item: ConversationItem

class ConversationItemDelete(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ConversationItemTruncate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int = 0
    audio_end_ms: int

class ConversationItemInputAudioTranscriptionComplete(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.input_audio_transcription.completed"] = "conversation.item.input_audio_transcription.completed"
    item_id: str
    content_index: int = 0
    transcript: str



class ResponseCreate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["response.create"] = "response.create"
    response: Optional[SessionConfig] = Field(description="Optional session config object to overwrite the session config for this response", default=None)


class ResponseCancel(BaseModel):
    event_id: Optional[str] = None
    type: Literal["response.cancel"]
    response_id: Optional[str] = None

class SessionCreated(BaseModel):
    event_id: Optional[str] = None
    type: Literal["session.created"]
    session: SessionConfig


class SessionUpdate(BaseModel):
    event_id: Optional[str] = None
    type: Literal["session.update"]
    session: SessionConfig




