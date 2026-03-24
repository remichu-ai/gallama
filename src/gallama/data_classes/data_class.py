from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator, HttpUrl, conint, root_validator
from typing import Optional, Literal, List, Dict, Union, Any, Type, Callable, Set
import asyncio
import os
import uuid
import time
import torch
import re
import base64
from datetime import datetime, timezone
from PIL import Image
from ..logger import logger
from .video import VideoFrame
from ..remote_mcp.models import MCPServerConfig


class TagEqualityMixin:
    def __eq__(self, other):
        # Check if the other object has a 'tag_type' attribute
        if hasattr(other, 'tag_type'):
            return self.tag_type == other.tag_type
        # If not, return NotImplemented to let Python handle the mismatch
        # (or try the other object's __eq__)
        return NotImplemented

def _default_post_processor(text:str, extra_args=None, **kwargs) -> str:
    return text


class TagDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    # FIX: Type must be Optional if default is None
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None

    marker_type: Literal["string", "regex"] = Field(
        description="Type of regex pattern",
        default="string"
    )

    include_markers: bool = Field(
        description="If True, the start and end tags (e.g. <tag>...</tag>) are included in the output string.",
        default=False
    )

    tag_type: str = "text"
    wait_till_complete: bool = Field(
        description="Wait until tag complete or switch to another tag",
        default=False
    )

    prompt_init: Optional[Callable[[Any], str]] = Field(
        description="optional function to get the string to init the prompt, currently used for tool prompting",
        exclude=True,
        default=None
    )

    post_processor: Callable[[str, Optional[Dict]], Any] = Field(
        description="optional function to post process the input",
        exclude=True,
        default=_default_post_processor
    )

    api_tag: str = Field(
        description="The tag to be used when returning to client",
        default="content"
    )
    role: Literal["system", "user", "assistant", "tool"] = "assistant"

    # FIX: Type must be Optional because default is None.
    # Logic to ensure it is populated happens in the validator.
    allowed_roles: Optional[Set[str]] = Field(
        description="For tag that can take on different roles depending on context",
        default=None
    )

    def __eq__(self, other):
        if hasattr(other, 'tag_type'):
            return self.tag_type == other.tag_type
        return False

    # FIX: Combined V1 validators into a single V2 model_validator
    @model_validator(mode='after')
    def validate_and_set_roles(self) -> 'TagDefinition':
        """
        1. Sets allowed_roles to {role} if it was not explicitly provided.
        2. Ensures that the value of the 'role' field is contained within 'allowed_roles'.
        """
        # Logic 1: Set Default
        if self.allowed_roles is None:
            self.allowed_roles = {self.role}

        # Logic 2: Validate consistency
        if self.role not in self.allowed_roles:
            raise ValueError(
                f"'role' value ('{self.role}') must be present in 'allowed_roles' ({self.allowed_roles})."
            )

        return self


class TextTag(TagEqualityMixin, BaseModel):
    tag_type: Literal["text"] = "text"


class GenericTag(BaseModel):
    tag_type: str


class Query(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class FunctionCall(BaseModel):
    arguments: str
    name: str

class ToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: str = "function"
    index: Optional[int] = None


class MultiModalTextContent(BaseModel):
    type: Literal["text"]
    text: str = ""

class MultiModalAudioContent(BaseModel):
    type: Literal["audio"]
    audio: str = ""     # base64 audio

class MultiModalImageContent(BaseModel):
    class ImageDetail(BaseModel):
        url: str = Field(description="URL, file path, or Data URI of the image")
        detail: Optional[Literal["low", "high"]] = "high"

        @validator('url')
        def validate_image_url(cls, v):
            if v.startswith('data:image/'):
                return cls.validate_data_uri(v)
            elif v.startswith('file://'):
                return cls.validate_local_file(v)
            elif v.startswith(('http://', 'https://')):
                return cls.validate_http_url(v)
            else:
                raise ValueError("Invalid image reference format")

        @classmethod
        def validate_data_uri(cls, v):
            data_uri_pattern = r'^data:image/(\w+);base64,(.+)$'
            match = re.match(data_uri_pattern, v)
            if not match:
                raise ValueError("Invalid Data URI format")

            image_format, base64_data = match.groups()
            try:
                decoded = base64.b64decode(base64_data)
                headers = {
                    'jpeg': b'\xff\xd8\xff',
                    'png': b'\x89PNG\r\n\x1a\n',
                    'gif': b'GIF87a',
                    'gif': b'GIF89a',
                    'webp': b'RIFF'
                }
                if not any(decoded.startswith(header) for header in headers.values()):
                    raise ValueError("Invalid image format")
            except base64.binascii.Error:
                raise ValueError("Invalid base64 string")
            return v

        @classmethod
        def validate_local_file(cls, v):
            file_path = v[7:]  # Remove 'file://' prefix
            if not os.path.isfile(file_path):
                raise ValueError("File does not exist")
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError("Invalid image file extension")
            return v

        @classmethod
        def validate_http_url(cls, v):
            try:
                HttpUrl(v)
            except ValueError:
                raise ValueError("Invalid HTTP URL")
            return v

    type: Literal["image_url"]
    image_url: Union[
        ImageDetail,        # openai spec
        str,                # huggingface
    ]

class MultiModalImageHFContent(BaseModel):
    type: Literal["image"]
    image_url: Union[
        str,                # huggingface - base64 string
        Image.Image,        # qwen, internal usage where the image object alrd obtained
    ]

    @validator('image_url')
    def validate_image(cls, value):
        if not isinstance(value, str) or not isinstance(value, Image.Image):
            raise ValueError("Not a valid PIL Image")
        return value


    class Config:
        arbitrary_types_allowed = True



class BaseMessage(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True
    )

    role: Literal['system', 'user', 'assistant', 'tool', 'developer']
    content: Optional[Union[
        str,
        List[Union[
            MultiModalTextContent,
            MultiModalImageContent,
            MultiModalAudioContent,
            MultiModalImageHFContent
        ]
    ]]] = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, description="previous reasoning for interleave thinking", alias="reasoning_content")

    @validator('content', pre=True)
    def convert_null_to_empty(cls, value):
        if value is None:
            return ""
        return value


class ParameterProperties(BaseModel):
    type: str = "object"
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    items: Optional[dict] = None   # for array


class ParameterSpec(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: List[str] = []


class FunctionSpec(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: ParameterSpec = None
    strict: bool = True     # not making any difference at the moment


class ToolSpec(BaseModel):
    type: str = "function"
    function: FunctionSpec


class MCPToolSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    authorization_token: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    allowed_tools: Optional[List[str]] = None
    require_approval: Optional[Union[str, Dict[str, Any]]] = None

    def to_mcp_server_config(self) -> MCPServerConfig:
        return MCPServerConfig(
            name=self.server_label,
            url=self.server_url,
            authorization_token=self.authorization_token,
            headers=self.headers,
            allowed_tools=self.allowed_tools,
            require_approval=self.require_approval,
        )


class SingleFunctionDict(BaseModel):
    name: str


class ToolForce(BaseModel):
    type: str = "function"
    function: SingleFunctionDict

    class Config:
        extra = "forbid"  # This will prevent extra keys in the dictionary

class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text", "json_object"]


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None

class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaSpec
    model_config = ConfigDict(extra="forbid")

class ChatMLQuery(BaseModel):
    # Configure the model to forbid extra fields
    # populate_by_name=True to support alias
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    class StreamOption(BaseModel):
        # differ to openai spec where it only return when set to true manually in stream mode
        include_usage: bool = False

    model: Optional[str] = ""
    messages: List[BaseMessage]
    temperature: Optional[float] = 0.7
    top_p: float = 0.85
    stream: Optional[bool] = False
    tools: Optional[List[Union[ToolSpec, MCPToolSpec]]] = None
    tool_choice: Union[None, Literal["none", "auto", "required"], ToolForce] = None
    tool_call_id: Optional[str] = None
    max_tokens: Optional[int] = Field(
        description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.",
        default=16000,
        alias="max_completion_tokens"
    )
    thinking_token_budget: Optional[int] = Field(
        default=None,
        description="Budget for the preliminary thinking/reasoning pass. Defaults to max(4096, max_tokens * 2)."
    )
    reasoning_effort: Optional[Literal[None, "minimal", "low", "medium", "high"]] = "medium"
    store: Optional[bool] = Field(
        description="Whether or not to store the output of this chat completion request for use in model distillation or evals products.",
        default=False,
    )
    # for video, currently for websocket
    video: Optional[List[Any]] = None    # TODO to have handling for list of base64 video frame

    # not part of openai api
    leading_prompt: Optional[str] = Field(default="", description="The string to append to the end of the prompt, this will not be part of the  generated response")
    prefix_strings: Optional[Union[str, List[str]]] = Field(default=None, description="String or list of strings to start the generation with. Can not be used together with regex_prefix_pattern")
    regex_pattern: Optional[constr(min_length=1)] = None   # regex to enforce
    regex_prefix_pattern: Optional[constr(min_length=1)] = Field(default=None, description="regex to enforce in the beginning of the generation, can not be used together with prefix_string")
    stop_words: Optional[List[str]] = Field(default=None, alias="stop")     # OpenAI use stop
    return_stop_word: Optional[bool] = False

    # tool call
    tool_call_thinking: bool = Field(default= True, description="Automatically trigger one liner tool call thinking when tool in auto mode to decide if tool is required")
    tool_call_thinking_token: int = Field(default= 200, description="Maximum token for tool thinking generation. If it exceed this threshold, no tool thinking is returned")
    tool_instruction_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the general instruction to use tool. prefix for best kv caching"))
    tool_schema_position: Literal["prefix", "postfix"] = (
        Field(default="postfix", description="Position of the schema of individual tools. If tool_schema is unchanged through out, "
                                            "keep it as prefix for maximum kv caching. postfix for cases where tool are changing between api request"))

    use_thinking: Literal[True, False, "Skip"] = Field(default=False, description="True to force thinking, False to do nothing, Skip to force skip")
    return_thinking: Optional[Literal[False, True, "separate"]] = Field(
        default=False,
        description="Return the generated thinking to front end. False - not return, True - return, 'separate' - return separately as .thinking field."
    )
    guided_decoding_backend: Optional[Literal["auto", "formatron", "lm-format-enforcer"]] = Field(
        default="auto",
        description="guided decoding backend. auto will choose the most suitable. If the selected backend is not working for specific llm backend (e.g. formatrong not working with llama cpp), selection will be auto"
    )

    # not yet supported options from here # TODO
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[int, float]] = {}
    top_logprob: int = None
    n: int = 1
    presence_penalty: float = 0
    response_format: Optional[Union[ResponseFormat, ResponseFormatJSONSchema]] = None
    seed: Optional[int] = None
    stream_options: Optional[StreamOption] = None
    parallel_tool_calls: bool = True            # default let the model handle and can not toggle

    def split_tools(self) -> tuple[List[ToolSpec], List[MCPServerConfig]]:
        function_tools: List[ToolSpec] = []
        mcp_servers: List[MCPServerConfig] = []

        for tool in self.tools or []:
            if isinstance(tool, MCPToolSpec):
                mcp_servers.append(tool.to_mcp_server_config())
            else:
                function_tools.append(tool)

        return function_tools, mcp_servers

    @validator('regex_pattern', 'regex_prefix_pattern')
    def validate_regex(cls, v):
        """ this function is to handle special regex patterns """
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f'Invalid regex pattern: {e}')
        return v

    @validator('tool_call_thinking_token')
    def validate_tool_call_thinking_token(cls, v):
        """ Validate that tool_call_thinking_token is greater than or equal to 0 """
        if v < 0:
            raise ValueError('tool_call_thinking_token must be greater than or equal to 0')
        return v

    @validator('thinking_token_budget')
    def validate_thinking_token_budget(cls, v):
        """Validate that thinking_token_budget is greater than or equal to 0 when provided."""
        if v is not None and v < 0:
            raise ValueError('thinking_token_budget must be greater than or equal to 0')
        return v

    @model_validator(mode='after')
    def set_default_thinking_token_budget(self):
        """Default the thinking token budget based on max_tokens when omitted."""
        if self.thinking_token_budget is None:
            max_tokens = self.max_tokens or 0
            self.thinking_token_budget = max(4096, max_tokens * 2)
        return self






# from here on is answer model for response to api request
class OneTool(BaseModel):
    """The format to use to call one tool"""
    name: str = Field(description='name of the function to use')
    arguments: str


class ToolCallResponse(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    function: OneTool
    type: Optional[str] = "function"


class ToolCalling(BaseModel):
    """ The format to call one or multiple tools """
    functions_calling: List[OneTool] = Field(description='the list of functions to call in chronological order',
                                             default=[])


class ChatResponse(BaseModel):
    role: Literal['system', 'user', 'assistant'] = None
    tool_call_id: Optional[str] = None
    content: Optional[str] = None
    # name: Optional[str] = None
    # function_call: Optional[ToolCalling] = None   # depreciated
    tool_calls: Optional[List[ToolCallResponse]] = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal['system', 'user', 'assistant', 'tool', 'developer'] = None
    tool_call_id: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    # thinking
    thinking: Optional[str] = None

    # function_call: Optional[ToolCalling] = None   # depreciated
    tool_calls: Optional[List[ToolCallResponse]] = None
    def __str__(self) -> str:
        if self.role == "system":
            return f"system:\n{self.content}\n"

        elif self.role == "function":
            return f"function name={self.name}:\n{self.content}\n"

        elif self.role == "user":
            if self.content is None:
                return "user:\n</s>"
            else:
                return f"user:\n</s>{self.content}\n"

        elif self.role == "assistant":
            if self.content is not None and self.function_call is not None:
                return f"assistant:\n{self.content}\nassistant to={self.function_call.name}:\n{self.function_call.arguments}</s>"

            elif self.function_call is not None:
                return f"assistant to={self.function_call.name}:\n{self.function_call.arguments}</s>"

            elif self.content is None:
                return "assistant"

            else:
                return f"assistant:\n{self.content}\n"
        else:
            raise ValueError(f"Unsupported role: {self.role}")


class UsageResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

OpenAIStopReason = Literal["stop", "length", "tool_calls", "content_filter"]
AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal", "model_context_window_exceeded"]

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    logprobs: Union[float, None] = None
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter"]
    ] = "stop"

    @classmethod
    def from_message(cls, message: ChatMessage, finish_reason: str):
        return cls(message=message, finish_reason=finish_reason)


class ChoiceDeltaToolCallFunction(BaseModel):
    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[str] = None


class ChoiceDelta(BaseModel):
    model_config = ConfigDict(extra='allow')

    content: Optional[str] = None
    #function_call: Optional[str] = None
    refusal: Optional[str] = None
    role: Optional[Literal[Literal['system', 'user', 'assistant', 'tool', None]]] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class StreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(description='id of the response', default_factory=lambda: "cmpl-" + str(uuid.uuid4().hex))
    object: Literal["chat.completion", "chat.completion.chunk"] = Field(default="chat.completion")
    created: int = Field(description='timestamp when object was created', default_factory=lambda: int(time.time()))
    model: str = Field(description='name of the model')
    choices: List[Union[Choice, StreamChoice]]
    usage: UsageResponse = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = Field(default="fp_44709d6fcb")
    choices: List[CompletionChoice]
    usage: Optional[UsageResponse] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    system_fingerprint: str = Field(default="fp_44709d6fcb")
    choices: List[CompletionChoice]


# embedding dataclass from here
class EmbeddingRequest(BaseModel):
    """ Request to embedding some text in the input"""
    input: Union[str, List[str], List[List[int]]] = Field(description="text to embed, a str or list of str")
    model: str
    dimension: Optional[int] = None
    encoding_format: Optional[Literal["float", "base64"]] = "float"


class EmbeddingObject(BaseModel):
    index: int = Field(description='index of the embedding', default=0)
    object: str = 'embedding'
    embedding: Union[List[float], str] = Field(description='list of float for embedding vector and str for base64')


class EmbeddingResponse(BaseModel):
    class Usage(BaseModel):
        prompt_tokens: int = Field(description='number of tokens in the prompt')
        total_tokens: int = Field(description='total number of tokens')

    object: str = 'list'
    model: str = Field(description='name of the model')
    usage: Usage
    data: List[EmbeddingObject]


class GenerateQuery(BaseModel):
    prompt: str = Field(description='prompt')
    model: Optional[str] = Field(description='name of the model', default=None)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(description='max number of tokens', default=None)


# class GenerationStats(BaseModel):
#     input_tokens_count: int = Field(description='input tokens count', default=0)
#     output_tokens_count: int = Field(description='output tokens count', default=0)
#     time_to_first_token: float = Field(description='time to first token', default=0)
#     time_generate: float = Field(description='time to generate tokens', default=0)
#
#     @property
#     def generation_speed(self) -> float:
#         if self.time_generate > 0:
#             return round(self.output_tokens_count / self.time_generate, ndigits=1)
#         else:
#             return 0
#
#     @property
#     def total_time(self) -> float:
#         return round(self.time_to_first_token + self.time_generate, ndigits=1)
#
#     @property
#     def prefill_speed(self) -> float:
#         if self.time_to_first_token > 0:
#             return round(self.input_tokens_count / self.time_to_first_token, ndigits=1)
#         else:
#             return 0

    # @property
    # def total_tokens_count(self) -> float:
    #     return self.input_tokens_count + self.output_tokens_count


class ModelObject(BaseModel):
    id: str = Field(description='id of the model')
    object: str = Field(description='object type', default="model")
    owned_by: str = Field(description='object owner', default="remichu")
    created_by: int = Field(description='model creation time', default=1686935002)


class ModelObjectResponse(BaseModel):
    object: str = Field(description='object type', default="list")
    data: List[ModelObject] = []


def _default_anthropic_model_created_at() -> str:
    return datetime.fromtimestamp(1686935002, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class AnthropicModelObject(BaseModel):
    id: str = Field(description="id of the model")
    type: Literal["model"] = "model"
    display_name: str = Field(description="display name of the model")
    created_at: str = Field(
        description="RFC 3339 datetime string indicating when the model was created",
        default_factory=_default_anthropic_model_created_at,
    )


class AnthropicModelListResponse(BaseModel):
    data: List[AnthropicModelObject] = Field(default_factory=list)
    first_id: Optional[str] = None
    has_more: bool = False
    last_id: Optional[str] = None


class Thinking(BaseModel):
    xml: str = Field(description='xml string')
    regex: str = Field(description='regex string to enforce any value', default=None)


class VoiceConfig(BaseModel):
    """Configuration for a single voice sample"""
    ref_audio_path: str = Field(description='path to the sound sample')
    ref_audio_transcription: str = Field(description='text of the voice sample')
    language: str = Field(description='language of the voice sample')
    speed_factor: Optional[float] = Field(description='speed factor of the voice sample', default=1.0)

    model_config = ConfigDict(extra="forbid")



# list of supported backend. None meaning it the api will take backend set from yaml config file
SUPPORTED_BACKENDS = [
    "exllama",
    "exllamav3",
    "vllm",
    "llama_cpp",
    "llama_cpp_server",
    "ik_llama",
    "transformers",
    "mlx_vlm",
    "sglang",
    "embedding",
    "faster_whisper",
    "mlx_whisper",
    "kokoro",
    None
]

class ModelSpec(BaseModel):
    model_id: Optional[str] = Field(description='id of the model which should be the path to the model', default=None)
    model_name: Optional[str] = Field(description='name of the model, which is the key inside yml configuration file', default=None)
    model_type: Optional[Literal["stt", "llm", "tts", "embedding", None]] = Field(description='type of the model, will be automatically determined based on backend', default=None)
    gpus: Optional[Union[Literal["auto"], List[float]]] = Field(description='VRam usage for each GPU', default="auto")
    cache_size: Optional[int] = Field(default=None, description='The context length for cache text in int. If None, will be set to the model context length')
    cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default="FP16", description='the quantization to use for cache, will use Q4 if not specified')
    max_seq_len: Optional[int] = Field(description="max sequence length", default=None)
    backend: Optional[Union[Literal[tuple(SUPPORTED_BACKENDS)], None]] = Field(description="model engine backend", default=None)
    tensor_parallel: Optional[bool] = Field(description="tensor parallel mode", default=False)
    prompt_template: Optional[str] = Field(description="prompt template", default=None)
    eos_token_list: List[str] = Field(description="eos tokens, can customize token here", default_factory=list)

    quant: Optional[float] = Field(description="quantization if the model support quantization on the fly", default=None)

    # number of concurrent request this model can handle
    max_concurrent_requests: int = Field(description="number of concurrent request this model can handle", default=1)

    # extra argument for specific backend or model
    backend_extra_args: Optional[Dict[Any, Any]] = Field(description="extra args to pass to the backend", default_factory=dict)

    # speculative decoding
    draft_model_id: Optional[str] = Field(description='id of the draft model', default=None)
    draft_model_name: Optional[str] = Field(description='name of the draft model', default=None)
    draft_gpus: Optional[Union[Literal["auto"], List[float]]] = Field(description='VRam usage for each GPU', default="auto")
    draft_cache_size: Optional[int] = Field(description='The context length for cache text in int. If None, will be set to the model context length', default=None)
    draft_cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default="FP16", description='the quantization to use for cache, will use Q4 if not specified')
    # backend is assumed to be the same as main model


    # whether require api call to match model name or not
    strict: bool = Field(description="whether require api call to match model name or not", default=False)

    # argument for different voice sample
    voice: Optional[Dict[str, VoiceConfig]] = Field(
        default=None,
        description="Voice configurations mapping voice names to their settings"
    )

    # audio related setting
    language: Optional[str] = Field(description="language of the audio", default="auto")

    # dont allow non recognizable option
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())  # disable protected_namespaces due to it field use model_ in the name


    @validator('gpus', pre=True, always=True)
    def validate_gpus(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            # Convert the dict to a list based on GPU IDs
            return [v.get(i, 0.0) for i in range(torch.cuda.device_count())]
        return v

    # TODO this is clasing with embedding cause embedding will set visiable GPU and hence it is not seen anymore in this validator
    # @validator('gpus')
    # def check_gpus(cls, gpus):
    #     if gpus is None:
    #         return None
    #     num_gpus = torch.cuda.device_count()
    #     for gpu_id, vram in enumerate(gpus):
    #         if gpu_id < 0 or gpu_id >= num_gpus:
    #             raise ValueError(f"Invalid GPU ID {gpu_id}. Must be between 0 and {num_gpus - 1}")
    #
    #         if vram < 0:
    #             raise ValueError(f"VRAM usage for GPU {gpu_id} must be a non-negative number")
    #
    #         if vram > 0:
    #             device = torch.cuda.get_device_properties(gpu_id)
    #             total_vram = device.total_memory / (1024 ** 3)  # Convert bytes to GB
    #             if vram > total_vram:
    #                 raise ValueError(
    #                     f"Requested VRAM ({vram} GB) for GPU {gpu_id} exceeds available VRAM ({total_vram:.2f} GB)")

        # return gpus


    @classmethod
    def get_model_type_from_backend(cls, backend: str = None):
        if backend is None:
            return None
        elif backend in ["exllama", "llama_cpp", "llama_cpp_server", "ik_llama", "transformers", "mlx_vlm", "sglang", "exllamav3", "vllm"]:
            return "llm"
        elif backend in ["faster_whisper", "mlx_whisper"]:
            return "stt"
        elif backend in ["kokoro"]:
            return "tts"
        elif backend in ["embedding"]:
            return "embedding"

    @classmethod
    def from_dict(cls, input_data: Union[str, Dict[str, Any]]):
        if isinstance(input_data, str):
            # If input is a string, split it into a dictionary
            params = input_data.split()
            input_dict = {}
            for param in params:
                key, value = param.split('=')
                input_dict[key] = value.strip("'")  # Strip single quotes here as well
        else:
            # If input is already a dictionary, use it as is
            input_dict = input_data

        model_id = input_dict.get('model_id')
        if model_id:
            model_id = input_dict.get('model_id').strip("'")  # Remove single quotes if present
        model_name = input_dict.get('model_name')
        max_seq_len = input_dict.get('max_seq_len', None)
        gpus = input_dict.get('gpus')
        cache_size = input_dict.get('cache_size')
        backend = input_dict.get('backend', None)  # Default to None if not provided
        tensor_parallel = input_dict.get('tp', False)
        if tensor_parallel=="True" or tensor_parallel=="true":
            tensor_parallel = True

        # TODO clean up code and merge config setting into config manager
        if backend == "None":
            backend = None
        cache_quant = input_dict.get('cache_quant', None)

        if gpus and isinstance(gpus, str) and gpus.lower() != "auto":
            gpus = [float(x) for x in gpus.split(',')]

        if cache_size:
            cache_size = int(cache_size)

        strict = input_dict.get('strict', False)
        voice = input_dict.get('voice', {})

        model_type = input_dict.get('model_type', None)
        if model_type is None:
            model_type = cls.get_model_type_from_backend(backend)

        prompt_template = input_dict.get('prompt_template', None)

        # concurrent request
        allowed_concurrency = 50 if backend in ["exllama", "exllamav3", "embedding"] else 1  # TODO to look into optimal number for each backend
        max_concurrent_requests = input_dict.get('max_concurrent_requests', allowed_concurrency)

        backend_extra_args = input_dict.get('backend_extra_args', {})

        # speculative decoding
        draft_model_id = input_dict.get('draft_model_id')
        if draft_model_id:
            draft_model_id = draft_model_id.strip("'")  # Remove single quotes if present

        draft_model_name = input_dict.get('draft_model_name', None)

        draft_gpus = input_dict.get('draft_gpus')
        draft_cache_size = input_dict.get('draft_cache_size')
        draft_cache_quant = input_dict.get('draft_cache_quant', None)

        if draft_gpus and isinstance(draft_gpus, str) and draft_gpus.lower() != "auto":
            draft_gpus = [float(x) for x in draft_gpus.split(',')]

        if draft_cache_size:
            draft_cache_size = int(draft_cache_size)

        return cls(model_id=model_id, model_name=model_name, model_type=model_type,
                   gpus=gpus, cache_size=cache_size, backend=backend, cache_quant=cache_quant,
                   strict=strict,
                   voice=voice,
                   prompt_template=prompt_template,
                   backend_extra_args=backend_extra_args,
                   max_concurrent_requests=max_concurrent_requests,
                   max_seq_len=max_seq_len,
                   tensor_parallel=tensor_parallel,
                   draft_model_id=draft_model_id, draft_model_name=draft_model_name,
                   draft_gpus=draft_gpus, draft_cache_size=draft_cache_size, draft_cache_quant=draft_cache_quant)

    def get_visible_gpu_indices(self) -> str:
        """
        Generate a string of GPU indices based on allocated GPUs.
        If no GPUs are specified, return all available GPU indices.
        Respects existing CUDA_VISIBLE_DEVICES environment variable if set.
        """
        import os

        # 1. First, check if the user explicitly set the env variable externally
        existing_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')

        if self.gpus is None or self.gpus == "auto":
            if existing_cvd is not None:
                # Pass through the exact string provided via CLI (e.g., "2,3,0,1,4,5")
                return existing_cvd
            else:
                # Fallback if launched without CLI variables
                import torch
                return ','.join(str(i) for i in range(torch.cuda.device_count()))

        if all(vram == 0 for vram in self.gpus):
            return ""  # No GPUs allocated

        visible_devices = [str(i) for i, vram in enumerate(self.gpus) if vram > 0]
        return ','.join(visible_devices)

    @staticmethod
    def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, handling nested dictionaries.
        Values from dict2 take precedence over dict1 except for nested dictionaries,
        which are merged recursively.

        Args:
            dict1 (Dict[str, Any]): First dictionary
            dict2 (Dict[str, Any]): Second dictionary (takes precedence for non-dict values)

        Returns:
            Dict[str, Any]: Merged dictionary
        """
        merged = dict1.copy()

        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # If both values are dictionaries, merge them recursively
                merged[key] = ModelSpec.deep_merge_dicts(merged[key], value)
            else:
                # For non-dict values or when key doesn't exist in dict1,
                # take the value from dict2
                merged[key] = value

        return merged

    def merge_with_config(self, model_config: Dict[str, Any]) -> 'ModelSpec':
        """
        Merges the current ModelSpec instance with a model configuration dictionary.
        Values from the current ModelSpec take precedence over the model_config.
        Handles nested dictionaries by merging them recursively.

        Args:
            model_config (Dict[str, Any]): Configuration dictionary from config manager

        Returns:
            ModelSpec: A new ModelSpec instance with merged configurations
        """
        # Only preserve fields explicitly provided by the caller. Otherwise
        # ModelSpec defaults like tensor_parallel=False overwrite values coming
        # from model_config.yaml during merge.
        spec_dict = self.model_dump(exclude_none=True, exclude_unset=True)

        # Merge configurations recursively
        merged_config = ModelSpec.deep_merge_dicts(model_config, spec_dict)

        # Create new ModelSpec instance with merged configuration
        return ModelSpec(**merged_config)

    @classmethod
    def from_merged_config(cls, model_spec: 'ModelSpec', model_config: Dict[str, Any]) -> 'ModelSpec':
        """
        Class method to create a new ModelSpec instance by merging an existing ModelSpec
        with a model configuration dictionary.

        Args:
            model_spec (ModelSpec): Existing ModelSpec instance
            model_config (Dict[str, Any]): Configuration dictionary from config manager

        Returns:
            ModelSpec: A new ModelSpec instance with merged configurations
        """
        return model_spec.merge_with_config(model_config)



class ModelDownloadSpec(BaseModel):
    """ dataclass for model download"""
    model_name: str
    quant: Optional[float] = None
    backend: Literal[tuple(SUPPORTED_BACKENDS)] = None

    # disable protected_namespaces due to it field use model_ in the name
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# === Anthropic/Claude Messages API Data Classes ===

class AnthropicTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicBase64ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class AnthropicURLImageSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


class AnthropicImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: Union[AnthropicBase64ImageSource, AnthropicURLImageSource] = Field(discriminator="type")


class AnthropicToolUseContent(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicToolResultContent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[AnthropicTextContent]]


class AnthropicMCPToolUseContent(BaseModel):
    type: Literal["mcp_tool_use"] = "mcp_tool_use"
    id: str
    name: str
    server_name: str
    input: Dict[str, Any]


class AnthropicMCPToolResultContent(BaseModel):
    type: Literal["mcp_tool_result"] = "mcp_tool_result"
    tool_use_id: str
    is_error: bool = False
    content: Union[str, List[AnthropicTextContent]]


class AnthropicToolInputSchema(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: List[str] = []


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: AnthropicToolInputSchema
    strict: Optional[bool] = None


class AnthropicMCPServer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["url"] = "url"
    url: str
    name: str
    authorization_token: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    def to_mcp_server_config(self, allowed_tools: Optional[List[str]] = None) -> MCPServerConfig:
        return MCPServerConfig(
            name=self.name,
            url=self.url,
            authorization_token=self.authorization_token,
            headers=self.headers,
            allowed_tools=allowed_tools,
        )


class AnthropicMCPToolset(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["mcp_toolset"] = "mcp_toolset"
    mcp_server_name: str
    default_config: Optional[Dict[str, Any]] = None
    configs: Optional[List[Any]] = None
    allowed_tools: Optional[List[str]] = None

    def get_allowed_tool_names(self) -> Optional[List[str]]:
        names: List[str] = []

        for value in self.allowed_tools or []:
            if value:
                names.append(str(value))

        for config in self.configs or []:
            if isinstance(config, str):
                names.append(config)
            elif isinstance(config, dict):
                for key in ("name", "tool_name"):
                    if config.get(key):
                        names.append(str(config[key]))
                        break

        return names or None


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "none", "tool"]
    name: Optional[str] = None  # only when type="tool"


class AnthropicTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class AnthropicThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = "NA"

class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicMCPToolUseBlock(BaseModel):
    type: Literal["mcp_tool_use"] = "mcp_tool_use"
    id: str
    name: str
    server_name: str
    input: Dict[str, Any]


class AnthropicMCPToolResultBlock(BaseModel):
    type: Literal["mcp_tool_result"] = "mcp_tool_result"
    tool_use_id: str
    is_error: bool = False
    content: List[AnthropicTextBlock]


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[
        AnthropicTextContent,
        AnthropicThinkingBlock,
        AnthropicImageContent,
        AnthropicToolUseContent,
        AnthropicToolResultContent,
        AnthropicMCPToolUseContent,
        AnthropicMCPToolResultContent,
    ]]]

class AnthropicOutputFormat(BaseModel):
    type: Literal["json_schema"]
    schema_: Dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class AnthropicOutputConfig(BaseModel):
    format: Optional[AnthropicOutputFormat] = None
    effort: Optional[Literal["low", "medium", "high", "max"]] = None


class AnthropicMessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: "msg_" + uuid.uuid4().hex)
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[
        AnthropicTextBlock,
        AnthropicThinkingBlock,
        AnthropicToolUseBlock,
        AnthropicMCPToolUseBlock,
        AnthropicMCPToolResultBlock,
    ]]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int  # required in Claude API
    system: Optional[Union[str, List[AnthropicTextContent]]] = None
    strip_claude_code_billing_header: bool = True
    tools: Optional[List[Union[AnthropicTool, AnthropicMCPToolset]]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[dict] = None
    output_config: Optional[AnthropicOutputConfig] = None
    mcp_servers: Optional[List[AnthropicMCPServer]] = None

    @staticmethod
    def _is_claude_code_billing_header_text(text: str) -> bool:
        if not text:
            return False

        text_normalized = text.strip()
        return (
            text_normalized.startswith("x-anthropic-billing-header:")
            and "cc_version=" in text_normalized
            and "cch=" in text_normalized
        )

    def remove_claude_code_billing_header_system_message(self) -> bool:
        """
        Remove Claude Code's volatile billing header from the top-level system
        prompt so it does not spoil prompt caching across turns.
        """
        if not self.system:
            return False

        if isinstance(self.system, str):
            kept_lines = [
                line for line in self.system.splitlines()
                if not self._is_claude_code_billing_header_text(line)
            ]
            cleaned_system = "\n".join(kept_lines).strip()
            changed = cleaned_system != self.system.strip()
            self.system = cleaned_system or None
            return changed

        cleaned_blocks = [
            block for block in self.system
            if not (
                getattr(block, "type", "") == "text"
                and self._is_claude_code_billing_header_text(block.text)
            )
        ]
        changed = len(cleaned_blocks) != len(self.system)
        self.system = cleaned_blocks or None
        return changed

    def get_ChatMLQuery(self) -> ChatMLQuery:
        import json
        chat_messages = []

        # 1. Translate Top-Level System Prompt to System Message
        if self.system:
            sys_content = ""
            if isinstance(self.system, str):
                sys_content = self.system
            elif isinstance(self.system, list):
                # Concatenate all text blocks if system prompt is a list
                sys_content = "\n".join([b.text for b in self.system if getattr(b, "type", "") == "text"])

            chat_messages.append(BaseMessage(role="system", content=sys_content))

        # 2. Translate Messages and Content Blocks
        for msg in self.messages:
            if isinstance(msg.content, str):
                chat_messages.append(BaseMessage(role=msg.role, content=msg.content))
            else:
                text_and_media_contents = []
                tool_calls = []
                tool_results = []
                reasoning_text = None

                for block in msg.content:
                    if block.type == "text":
                        text_and_media_contents.append(
                            MultiModalTextContent(type="text", text=block.text)
                        )

                    elif block.type == "thinking":
                        # Extract Anthropic thinking block into BaseMessage's reasoning field
                        if reasoning_text is None:
                            reasoning_text = block.thinking
                        else:
                            reasoning_text += "\n" + block.thinking

                    elif block.type == "image":
                        # Convert Anthropic Image to OpenAI Image URL Format
                        if getattr(block.source, "type", None) == "base64":
                            data_uri = f"data:{block.source.media_type};base64,{block.source.data}"
                            img_detail = MultiModalImageContent.ImageDetail(url=data_uri)
                            text_and_media_contents.append(
                                MultiModalImageContent(type="image_url", image_url=img_detail)
                            )
                        elif getattr(block.source, "type", None) == "url":
                            img_detail = MultiModalImageContent.ImageDetail(url=block.source.url)
                            text_and_media_contents.append(
                                MultiModalImageContent(type="image_url", image_url=img_detail)
                            )

                    elif block.type == "tool_use":
                        # Extract Anthropic tool usage into OpenAI tool_calls
                        func_call = FunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input)
                        )
                        tool_calls.append(ToolCall(id=block.id, function=func_call, type="function"))

                    elif block.type == "mcp_tool_use":
                        func_call = FunctionCall(
                            name=f"mcp__{block.server_name}__{block.name}",
                            arguments=json.dumps(block.input)
                        )
                        tool_calls.append(ToolCall(id=block.id, function=func_call, type="function"))

                    elif block.type == "tool_result":
                        # Anthropic passes tool results inside a "user" message.
                        # OpenAI requires these to be separate "tool" role messages.
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            content_str = "\n".join([b.text for b in block.content if getattr(b, "type", "") == "text"])

                        tool_results.append(BaseMessage(
                            role="tool",
                            tool_call_id=block.tool_use_id,
                            content=content_str
                        ))

                    elif block.type == "mcp_tool_result":
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            content_str = "\n".join([b.text for b in block.content if getattr(b, "type", "") == "text"])

                        tool_results.append(BaseMessage(
                            role="tool",
                            tool_call_id=block.tool_use_id,
                            content=content_str
                        ))

                # Handling the assembled blocks
                if msg.role == "user" and tool_results:
                    # If the user message had text along with the tool results, add it first
                    if text_and_media_contents:
                        chat_messages.append(BaseMessage(role="user", content=text_and_media_contents))
                    # Then append all the separated tool result messages
                    chat_messages.extend(tool_results)
                else:
                    # Standard message (User or Assistant)
                    base_msg = BaseMessage(
                        role=msg.role,
                        content=text_and_media_contents if text_and_media_contents else "",
                        tool_calls=tool_calls if tool_calls else None,
                        reasoning=reasoning_text
                    )
                    chat_messages.append(base_msg)

        # 3. Translate Tools Schema
        chat_tools = None
        if self.tools:
            chat_tools = []
            for t in self.tools:
                if isinstance(t, AnthropicMCPToolset):
                    continue
                param_spec = ParameterSpec(
                    type=t.input_schema.type,
                    properties=t.input_schema.properties,
                    required=t.input_schema.required
                )
                func_spec = FunctionSpec(
                    name=t.name,
                    description=t.description,
                    parameters=param_spec
                )
                chat_tools.append(ToolSpec(type="function", function=func_spec))
            if not chat_tools:
                chat_tools = None

        # 4. Translate Tool Choice Constraints
        chat_tool_choice = None
        if self.tool_choice:
            if self.tool_choice.type == "auto":
                chat_tool_choice = "auto"
            elif self.tool_choice.type == "any":
                chat_tool_choice = "required"
            elif self.tool_choice.type == "none":
                chat_tool_choice = "none"
            elif self.tool_choice.type == "tool" and self.tool_choice.name:
                chat_tool_choice = ToolForce(
                    type="function",
                    function=SingleFunctionDict(name=self.tool_choice.name)
                )

        # 4b. Translate output_config to response_format
        chat_response_format = None
        chat_reasoning_effort = None
        if self.output_config and self.output_config.format:
            fmt = self.output_config.format
            if fmt.type == "json_schema":
                chat_response_format = ResponseFormatJSONSchema(
                    type="json_schema",
                    json_schema=JsonSchemaSpec(
                        name="anthropic_structured_output",
                        schema=fmt.schema_,
                        strict=True
                    )
                )
        if self.output_config and self.output_config.effort:
            # Gallama's internal schema does not distinguish Anthropic "max"
            # from "high", so map it to the closest supported value.
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "max": "high",
            }
            chat_reasoning_effort = effort_map[self.output_config.effort]

        # 5. Build and return the final ChatMLQuery object
        query_kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "tools": chat_tools,
            "tool_choice": chat_tool_choice,
            "stop_words": self.stop_sequences,
            "response_format": chat_response_format,
            "reasoning_effort": chat_reasoning_effort,
        }

        # Filter out None values so Pydantic applies its own defaults
        filtered_kwargs = {k: v for k, v in query_kwargs.items() if v is not None}

        logger.debug(f"converted to ChatMLQuery: {filtered_kwargs}")

        return ChatMLQuery(**filtered_kwargs)

    def get_mcp_server_configs(self) -> List[MCPServerConfig]:
        if not self.mcp_servers:
            return []

        toolsets_by_server = {
            toolset.mcp_server_name: toolset
            for toolset in self.tools or []
            if isinstance(toolset, AnthropicMCPToolset)
        }

        configs: List[MCPServerConfig] = []
        for server in self.mcp_servers:
            toolset = toolsets_by_server.get(server.name)
            if toolset is None:
                continue
            configs.append(server.to_mcp_server_config(allowed_tools=toolset.get_allowed_tool_names()))

        return configs
