from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator, HttpUrl, conint
from typing import Optional, Literal, List, Dict, Union, Any, Type
import asyncio
import os
import uuid
import time
import torch
import re
import base64


class TextTag(BaseModel):
    tag_type: Literal["text"] = "text"


class ArtifactTag(BaseModel):
    tag_type: Literal["artifact"] = "artifact"
    artifact_type: Literal["code", "self_contained_text"]
    identifier: str
    title: str
    language: Optional[str] = None


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
    image_url: ImageDetail


class BaseMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[Union[str, List[Union[MultiModalTextContent, MultiModalImageContent]]]] = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

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


class SingleFunctionDict(BaseModel):
    name: str


class ToolForce(BaseModel):
    type: str = "function"
    function: SingleFunctionDict

    class Config:
        extra = "forbid"  # This will prevent extra keys in the dictionary


# test_thinking = """
# <plan>
#   <task>Brief task summary</task>
#   <structure>
#     <c1>[text]: Acknowledge question and introduce answer</c1>
#     <c2>[artifact]: Main content (e.g., code)</c2>
#     <c3>[text]: Explain or elaborate on c2</c3>
#     <c4>[artifact]: Additional content if needed</c4>
#     <c5>[text]: Explain or elaborate on c4</c5>
#     <!-- Add more pairs if needed -->
#   </structure>
# </plan>
# """

class ChatMLQuery(BaseModel):
    # Configure the model to forbid extra fields
    model_config = ConfigDict(extra="forbid")

    class ResponseFormat(BaseModel):
        type: Literal["text", "json_object"]

    class StreamOption(BaseModel):
        include_usage: bool = False

    model: Optional[str] = ""
    messages: List[BaseMessage]
    temperature: Optional[float] = 0.01
    top_p: float = 0.85
    stream: Optional[bool] = False
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Union[None, Literal["none", "auto", "required"], ToolForce] = None
    tool_call_id: Optional[str] = None
    max_tokens: Optional[int] = None

    # not part of openai api
    leading_prompt: Optional[str] = Field(default="", description="The string to append to the end of the prompt, this will not be part of the  generated response")
    prefix_strings: Optional[Union[str, List[str]]] = Field(default=None, description="String or list of strings to start the generation with. Can not be used together with regex_prefix_pattern")
    regex_pattern: Optional[constr(min_length=1)] = None   # regex to enforce
    regex_prefix_pattern: Optional[constr(min_length=1)] = Field(default=None, description="regex to enforce in the beginning of the generation, can not be used together with prefix_string")
    stop_words: Optional[List[str]] = Field(default=None, alias="stop")     # OpenAI use stop
    thinking_template: Optional[str] = None

    # tool call
    tool_call_thinking: bool = Field(default= True, description="Automatically trigger one liner tool call thinking when tool in auto mode to decide if tool is required")
    tool_call_thinking_token: int = Field(default= 200, description="Maximum token for tool thinking generation. If it exceed this threshold, no tool thinking is returned")
    tool_instruction_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the general instruction to use tool. prefix for best kv caching"))
    tool_schema_position: Literal["prefix", "postfix"] = (
        Field(default="postfix", description="Position of the schema of individual tools. If tool_schema is unchanged through out, "
                                            "keep it as prefix for maximum kv caching. postfix for cases where tool are changing between api request"))

    artifact: Optional[Literal["No", "Fast", "Slow"]] = Field(default="No", description="Normal will parse the streamed output for artifact, whereas Strict is slower and will use format enforcer to enforce")
    return_thinking: Optional[Literal[False, True, "separate"]] = Field(
        default=False,
        description="Return the generated thinking to front end. False - not return, True - return, 'separate' - return separately as .thinking field. If used together with artifact, True will return as separate."
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
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stream_options: Optional[StreamOption] = None
    parallel_tool_calls: bool = True            # default let the model handle and can not toggle

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
    role: Literal['system', 'user', 'assistant'] = None
    tool_call_id: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    # thinking
    thinking: Optional[str] = None

    # function_call: Optional[ToolCalling] = None   # depreciated
    tool_calls: Optional[List[ToolCallResponse]] = None
    artifact_meta: Union[TextTag, ArtifactTag] = None

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
    content: Optional[str] = None
    function_call: Optional[str] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
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

    @property
    def total_tokens_count(self) -> float:
        return self.input_tokens_count + self.output_tokens_count


class ModelObject(BaseModel):
    id: str = Field(description='id of the model')
    object: str = Field(description='object type', default="model")
    owned_by: str = Field(description='object owner', default="remichu")
    created_by: int = Field(description='model creation time', default=1686935002)


class ModelObjectResponse(BaseModel):
    object: str = Field(description='object type', default="list")
    data: List[ModelObject] = []


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
SUPPORTED_BACKENDS = ["exllama", "llama_cpp", "transformers", "embedding", "faster_whisper", "gpt_sovits", None]

class ModelSpec(BaseModel):
    model_id: Optional[str] = Field(description='id of the model which should be the path to the model', default=None)
    model_name: Optional[str] = Field(description='name of the model, which is the key inside yml configuration file', default=None)
    model_type: Optional[Literal["stt", "llm", "tts", "embedding", None]] = Field(description='type of the model, will be automatically determined based on backend', default=None)
    gpus: Optional[Union[Literal["auto"], List[float]]] = Field(description='VRam usage for each GPU', default="auto")
    cache_size: Optional[int] = Field(default=None, description='The context length for cache text in int. If None, will be set to the model context length')
    cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default=None, description='the quantization to use for cache, will use Q4 if not specified')
    max_seq_len: Optional[int] = Field(description="max sequence length", default=None)
    backend: Optional[Union[Literal[tuple(SUPPORTED_BACKENDS)], None]] = Field(description="model engine backend", default=None)
    tensor_parallel: Optional[bool] = Field(description="tensor parallel mode", default=False)
    prompt_template: Optional[str] = Field(description="prompt template", default=None)
    eos_token_list: List[str] = Field(description="eos tokens, can customize token here", default_factory=list)

    quant: Optional[float] = Field(description="quantization if the model support quantization on the fly", default=None)

    # number of concurrent request this model can handle
    max_concurrent_requests: int = Field(description="number of concurrent request this model can handle", default=1)

    # extra argument for specific backend or model
    backend_extra_args: Dict[Any, Any] = Field(description="extra args to pass to the backend", default_factory=dict)

    # speculative decoding
    draft_model_id: Optional[str] = Field(description='id of the draft model', default=None)
    draft_model_name: Optional[str] = Field(description='name of the draft model', default=None)
    draft_gpus: Optional[List[float]] = Field(description='VRam usage for each GPU', default=None)
    draft_cache_size: Optional[int] = Field(description='The context length for cache text in int. If None, will be set to the model context length', default=None)
    draft_cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default=None, description='the quantization to use for cache, will use Q4 if not specified')
    # backend is assumed to be the same as main model

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
        elif backend in ["exllama", "llama_cpp", "transformers"]:
            return "llm"
        elif backend in ["faster_whisper"]:
            return "stt"
        elif backend in ["gpt_sovits"]:
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

        if gpus:
            gpus = [float(x) for x in gpus.split(',')]

        if cache_size:
            cache_size = int(cache_size)

        if cache_size:
            cache_size = int(cache_size)

        model_type = input_dict.get('model_type', None)
        if model_type is None:
            model_type = cls.get_model_type_from_backend(backend)


        # concurrent request
        allowed_concurrency = 50 if backend in ["exllama", "embedding"] else 1  # TODO to look into optimal number for each backend
        max_concurrent_requests = input_dict.get('max_concurrent_requests', allowed_concurrency)

        backend_extra_args = input_dict.get('backend_extra_args', None)

        # speculative decoding
        draft_model_id = input_dict.get('draft_model_id')
        if draft_model_id:
            draft_model_id = draft_model_id.strip("'")  # Remove single quotes if present
        draft_model_name = input_dict.get('draft_model_name')
        if not draft_model_name and draft_model_id:
            draft_model_name = draft_model_id.split('/')[-1]
        else:
            draft_model_name = None

        draft_gpus = input_dict.get('draft_gpus')
        draft_cache_size = input_dict.get('draft_cache_size')
        draft_cache_quant = input_dict.get('draft_cache_quant', None)

        if draft_gpus:
            draft_gpus = [float(x) for x in draft_gpus.split(',')]

        if draft_cache_size:
            draft_cache_size = int(draft_cache_size)

        # Note: We don't need to set model_name here, as the validator will handle it
        return cls(model_id=model_id, model_name=model_name, model_type=model_type,
                   gpus=gpus, cache_size=cache_size, backend=backend, cache_quant=cache_quant,
                   max_concurrent_requests=max_concurrent_requests,
                   max_seq_len=max_seq_len,
                   tensor_parallel=tensor_parallel,
                   draft_model_id=draft_model_id, draft_model_name=draft_model_name,
                   draft_gpus=draft_gpus, draft_cache_size=draft_cache_size, draft_cache_quant=draft_cache_quant)

    def get_visible_gpu_indices(self) -> str:
        """
        Generate a string of GPU indices based on allocated GPUs.
        If no GPUs are specified, return all available GPU indices.

        Returns:
            str: A comma-separated string of GPU indices with allocated VRAM,
                 or all available GPU indices if none are specified.
        """
        if self.gpus is None or self.gpus == "auto":
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
        # Convert current ModelSpec to dict, excluding None values
        spec_dict = self.model_dump(exclude_none=True)

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
