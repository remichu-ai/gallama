from copy import deepcopy
from enum import Enum
from typing import Literal, Optional, Dict, List, Union

import numpy as np
from pydantic import BaseModel, Field

from gallama.data_classes.realtime_client_proto import Role



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


ConversationItemServer = Union[
    ConversationItemMessageServer,
    ConversationItemFunctionCallServer,
    ConversationItemFunctionCallOutputServer
]


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


class UsageResponseRealTime(BaseModel):

    class InputTokenDetail(BaseModel):
        class CachedToken(BaseModel):
            """this class was returned but not even mentioned in docs"""
            text_tokens: int = 0
            audio_tokens: int = 0

        cached_tokens: int = 0
        text_tokens: int = 0
        audio_tokens: int = 0
        cached_tokens_details: CachedToken = Field(default_factory=CachedToken)

    class OutputTokenDetail(BaseModel):
        text_tokens: int = 0
        audio_tokens: int = 0

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_token_details: InputTokenDetail = Field(default_factory=InputTokenDetail)
    output_token_details: OutputTokenDetail = Field(default_factory=OutputTokenDetail)

class ServerResponse(BaseModel):
    id: str
    object: Literal["realtime.response"] = "realtime.response"
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    status_details: Optional[Dict] = None
    output: Optional[List[ConversationItemServer]] = []
    usage: Optional[UsageResponseRealTime] = None


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