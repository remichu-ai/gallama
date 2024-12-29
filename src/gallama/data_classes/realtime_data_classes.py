from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum
from copy import deepcopy
import numpy as np


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
    type: Literal["server_vad"]
    threshold: Optional[float] = Field(ge=0.0, le=1.0,default=0.5)
    prefix_padding_ms: Optional[int] = Field(ge=0, default=500)
    silence_duration_ms: Optional[int] = Field(ge=0, default=700)
    create_response: bool = True

class SessionConfig(BaseModel):
    modalities: List[Literal["text", "audio"]] = Field(default_factory=lambda: ["text", "audio"])
    instructions: Optional[str] = ""    # system prompt
    voice: Optional[Voice] = None
    input_audio_format: Optional[AudioFormat] = None
    output_audio_format: Optional[AudioFormat] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    turn_detection: Optional[TurnDetectionConfig] = None
    tools: Optional[List[Tool]] = Field(default_factory=list)
    tool_choice: Optional[Union[ToolChoice, str]] = "auto"
    temperature: Optional[float] = Field(0.4, ge=0.1, le=1.2)
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = "inf"

    # extra
    model: Optional[str] = None

    # extra for gallama backend
    streaming_transcription: bool = True
    user_interrupt_token: Optional[str] = Field(description= "Custom word to insert everytime user interrupt the assistant",default=" <user_interrupt>")


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

# Define the discriminated union
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


# Server side event ####################################################################################


class ContentTypeServer(str, Enum):
    INPUT_TEXT = "input_text"
    INPUT_AUDIO = "input_audio"
    TEXT = "text"
    AUDIO = "audio"


class MessageContentServer(BaseModel):
    type: Literal["input_text", "input_audio", "text", "item_reference", "audio"]
    text: Optional[str] = None
    audio: Optional[np.ndarray] = None         # bytes audio, this is different to client class
    transcript: Optional[str] = None
    id: Optional[str] = None            # for item_reference type

    class Config:
        arbitrary_types_allowed = True


class ConversationItemBaseServer(BaseModel):
    id: Optional[str] = None
    object: Literal["realtime.item"] = "realtime.item"
    status: Optional[Literal["in_progress","completed", "incomplete"]] = None

    def strip_audio(self) -> "ConversationItemServer":
        """
        Creates a deep copy of the conversation item with audio fields removed from message content.

        Returns:
            A new conversation item with audio stripped from message content
        """
        # Create a deep copy
        stripped_item = deepcopy(self)

        # Only process message types that have content
        if isinstance(stripped_item, ConversationItemMessageServer):
            for content in stripped_item.content:
                content.audio = None
                # If this is an audio type message with no transcript,
                # provide empty string for text to ensure valid message
                if content.type == "audio" and not content.transcript:
                    content.text = ""

        return stripped_item


class ConversationItemMessageServer(ConversationItemBaseServer):
    type: Literal["message"] = "message"
    role: Role
    content: List[MessageContentServer]

class ConversationItemFunctionCallServer(ConversationItemBaseServer):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str

class ConversationItemFunctionCallOutputServer(ConversationItemBaseServer):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

# Define the discriminated union
ConversationItemServer = Union[
    ConversationItemMessageServer,
    ConversationItemFunctionCallServer,
    ConversationItemFunctionCallOutputServer
]
# Instead of using Union directly, create a function to discriminate the type
def parse_conversation_item(data: dict) -> ConversationItemBaseServer:
    item_type = data.get("type")
    if item_type == "message":
        return ConversationItemMessageServer(**data)
    elif item_type == "function_call":
        return ConversationItemFunctionCallServer(**data)
    elif item_type == "function_call_output":
        return ConversationItemFunctionCallOutputServer(**data)
    else:
        raise ValueError(f"Unknown item type: {item_type}")

class ConversationItemCreated(BaseModel):       # note that server event name is Created instead of Create
    event_id: Optional[str] = None
    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: Optional[str] = None
    item: ConversationItemServer

class ContentPart(BaseModel):
    type: Literal["text", "audio"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


# Server Response classes from here ###########################################
class UsageResponseRealTime(BaseModel):
    class CachedToken(BaseModel):
        text_tokens: int = 0
        audio_tokens: int = 0

    class InputTokenDetail(BaseModel):
        cached_tokens: int = 0
        text_tokens: int = 0
        audio_tokens: int = 0
        cached_tokens_details: int = 0

    class OutputTokenDetail(BaseModel):
        text_tokens: int = 0
        audio_tokens: int = 0

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_token_details: InputTokenDetail = InputTokenDetail()
    output_token_details: OutputTokenDetail = OutputTokenDetail()



class ServerResponse(BaseModel):
    id: str
    object: Literal["realtime.response"] = "realtime.response"
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    status_details: Optional[Dict] = None
    output: Optional[List[ConversationItemServer]] = []
    usage: Optional[Dict] = None



class ResponseCreated(BaseModel):
    """ this object is to send to client to signal that a response is created"""
    event_id: str
    type: Literal["response.created"] = "response.created"
    response: ServerResponse


class ResponseOutput_ItemAdded(BaseModel):
    event_id: str
    type: Literal["response.output_item.added"] = "response.output_item.added"
    response_id: str
    output_index: int
    item: ConversationItemServer


class ResponseContentPartAddedEvent(BaseModel):
    event_id: str
    type: Literal[
        "response.content_part.added",
        "response.content_part.done"
    ]
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: ContentPart


class ResponseContentPartDoneEvent(BaseModel):
    event_id: str
    type: Literal["response.content_part.done"] = "response.content_part.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: ContentPart

class ResponseDone(BaseModel):
    event_id: str
    type: Literal["response.done"] = "response.done"
    response_id: str
    response: ServerResponse


class ResponseDelta(BaseModel):
    event_id: str
    type: Literal["response.text.delta", "response.audio_transcript.delta", "response.audio.delta"]
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str

class ResponseTextDone(BaseModel):
    event_id: str
    type: Literal["response.text.done"] = "response.text.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    text: str


class ResponseAudioDone(BaseModel):
    event_id: str
    type: Literal["response.audio.done"] = "response.audio.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0

class ResponseTranscriptDone(BaseModel):
    event_id: str
    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    transcript: str


class ConversationItemDeleted(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    item_id: str

class ConversationItemTruncated(BaseModel):
    event_id: Optional[str] = None
    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    item_id: str
    content_index: int = 0
    audio_end_ms: int


class InputAudioBufferSpeechStarted(BaseModel):
    event_id: Optional[str] = None
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    audio_start_ms: int = Field(description="Milliseconds from the start of all audio written to the buffer during the session when speech was first detected. This will correspond to the beginning of audio sent to the model, and thus includes the prefix_padding_ms configured in the Session.")
    item_id: str

class InputAudioBufferSpeechStopped(BaseModel):
    event_id: Optional[str] = None
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    audio_end_ms: int = Field(description="Milliseconds since the session started when speech stopped. This will correspond to the end of audio sent to the model, and thus includes the min_silence_duration_ms configured in the Session.")
    item_id: str

# open ai actually doesnt return this event
# class ResponseCancelled(BaseModel):
#     event_id: Optional[str] = None
#     type: Literal["response.cancel"]
#     response_id: Optional[str] = None