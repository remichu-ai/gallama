from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum
from .audio_data_class import LanguageType


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
    prefix_padding_ms: Optional[int] = Field(ge=0, default=300)
    silence_duration_ms: Optional[int] = Field(ge=0, default=400)
    create_response: bool = True

    # gallama specific setting
    language: Optional[LanguageType] = ["en", "vi", "zh"]
    factor_prefix_padding_in_truncate: bool = Field(default=True,
                                                    description="Prefix padding will ensure speech start event only emitted "
                                                                "after certain ms of continuous speak, after which user will send conversation.item.truncate event"
                                                                "This setting is to automatically offset truncate timing by this amount of ms")


class VideoStreamSetting(BaseModel):
    video_stream: Optional[bool] = True
    # if video_max_resolution is None, there wont be any rescaling of image
    video_max_resolution: Literal["240p", "360p", "480p", "540p", "720p", "900p", "1080p", None] = "720p"
    retain_video: Optional[Literal["disable","message_based", "time_based"]] = Field(
        description="whether to retain images for past message", default="time_based")
    retain_per_message: int = Field(
        description="number of frame retained per message for old messages", default=1)
    second_per_retain: int = Field(
        description="one frame will be retained per this number of seconds", default=3)
    max_message_with_retained_video: int = Field(
        description="number of User messages that will have video frame retained", default=10)


class SessionConfig(BaseModel):
    modalities: List[Literal["text", "audio"]] = Field(default_factory=lambda: ["text", "audio"])
    instructions: Optional[str] = ""    # system prompt
    voice: Optional[str] = None
    input_audio_format: Optional[AudioFormat] = None
    output_audio_format: Optional[AudioFormat] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    turn_detection: Optional[TurnDetectionConfig] = Field(default_factory=TurnDetectionConfig)
    tools: Optional[List[Tool]] = Field(default_factory=list)
    tool_choice: Optional[Literal["auto", "none", "required"]] = "auto"
    temperature: Optional[float] = Field(0.4, ge=0.1, le=1.2)
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = "inf"

    # extra
    model: Optional[str] = None

    # extra for gallama backend
    streaming_transcription: bool = True
    user_interrupt_token: Optional[str] = Field(description= "Custom word to insert everytime user interrupt the assistant",default=" <user_interrupt>")
    input_sample_rate: Optional[int] = Field(description="Sample rate of input audio",default=24000)
    output_sample_rate: Optional[int] = Field(description="Sample rate of input audio",default=24000)

    # extra argument for gallama tool calling:
    tool_call_thinking: bool = Field(default= True, description="Automatically trigger one liner tool call thinking when tool in auto mode to decide if tool is required")
    tool_call_thinking_token: int = Field(default= 200, description="Maximum token for tool thinking generation. If it exceed this threshold, no tool thinking is returned")
    tool_instruction_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the general instruction to use tool. prefix for best kv caching"))
    tool_schema_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the schema of individual tools. If tool_schema is unchanged through out, "
                                            "keep it as prefix for maximum kv caching. postfix for cases where tool are changing between api request"))

    # extra argument for gallama video
    video: VideoStreamSetting = Field(default_factory=VideoStreamSetting)


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
        Properly handles nested Pydantic models like VideoStreamSetting and TurnDetectionConfig.

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
            other_config = other.copy() if other else {}

        # Recursively merge nested objects
        def recursive_merge(current, other):
            merged = current.copy()
            for key, value in other.items():
                if value is None:
                    continue

                if key not in merged:
                    merged[key] = value
                    continue

                current_value = merged[key]

                # Handle nested Pydantic models
                if isinstance(current_value, dict) and isinstance(value, dict):
                    merged[key] = recursive_merge(current_value, value)
                elif isinstance(value, dict) and isinstance(current_value, BaseModel):
                    # Convert current Pydantic model to dict and merge
                    current_dict = current_value.dict()
                    merged_dict = recursive_merge(current_dict, value)
                    # Create new instance of the same model type
                    merged[key] = current_value.__class__(**merged_dict)
                else:
                    merged[key] = value

            return merged

        # Perform the merge
        merged_config = recursive_merge(current_config, other_config)

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
    speech_start_time: Optional[float] = None
    speech_end_time: Optional[float] = None

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




