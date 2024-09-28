from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator
from typing import Optional, Literal, List, Dict, Union, Any, Type
import asyncio
import uuid
import time
import torch
import re


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


class ToolCall(BaseModel):
    class FunctionCall(BaseModel):
        arguments: str
        name: str

    id: str
    function: FunctionCall
    type: str = "function"
    index: Optional[int] = None


class BaseMessage(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[str] = ""
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
    class ResponseFormat(BaseModel):
        type: Literal["text", "json_object"]

    class StreamOption(BaseModel):
        include_usage: bool = False

    model: Optional[str] = "Mixtral-8x7B"
    messages: List[BaseMessage]
    temperature: Optional[float] = 0.01
    top_p: float = 0.85
    stream: Optional[bool] = False
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Union[None, Literal["none", "auto", "required"], ToolForce] = None
    tool_call_id: Optional[str] = None
    max_tokens: Optional[int] = None

    # not part of openai api
    leading_prompt: Optional[str] = Field(default="", description="The string to append to the end of the prompt, this will not be part of the generated response")
    prefix_strings: Optional[Union[str, List[str]]] = Field(default=None, description="String or list of strings to start the generation with. Can not be used together with regex_prefix_pattern")
    regex_pattern: Optional[constr(min_length=1)] = None   # regex to enforce
    regex_prefix_pattern: Optional[constr(min_length=1)] = Field(default=None, description="regex to enforce in the beginning of the generation, can not be used together with prefix_string")
    stop_words: Optional[List[str]] = Field(default=None, alias="stop")     # OpenAI use stop
    thinking_template: Optional[str] = None
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


class StreamChoice(BaseModel):
    index: int
    delta: ChatMessage = None
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls"]] = None


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


class GenerationStats(BaseModel):
    input_tokens_count: int = Field(description='input tokens count', default=0)
    output_tokens_count: int = Field(description='output tokens count', default=0)
    time_to_first_token: float = Field(description='time to first token', default=0)
    time_generate: float = Field(description='time to generate tokens', default=0)

    @property
    def generation_speed(self) -> float:
        if self.time_generate > 0:
            return round(self.output_tokens_count / self.time_generate, ndigits=1)
        else:
            return 0

    @property
    def total_time(self) -> float:
        return round(self.time_to_first_token + self.time_generate, ndigits=1)

    @property
    def prefill_speed(self) -> float:
        if self.time_to_first_token > 0:
            return round(self.input_tokens_count / self.time_to_first_token, ndigits=1)
        else:
            return 0

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

class GenStart(BaseModel):
    """ this item signal start of generation"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())  # disable protected_namespaces due to it field use model_ in the name

    gen_type: Literal["text", "tool", "thinking"]  = Field(description='True to signal end of generation', default="text")

class GenEnd(BaseModel):
    """ this item signal end of generation"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())  # disable protected_namespaces due to it field use model_ in the name

    generation_end: bool = Field(description='True to signal end of generation', default=True)


class GenText(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())  # disable protected_namespaces due to it field use model_ in the name

    text_type: Literal["text", "thinking", "tool"] = "text"
    content: str = Field(description='text or thinking')
    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, cls)


class GenQueue(asyncio.Queue):
    def __init__(self, maxsize=0, allowed_types: List[str] = [GenText, GenerationStats, GenStart, GenEnd]):
        super().__init__(maxsize)
        self.allowed_types = allowed_types

    def _check_item(self, item):
        if self.allowed_types:
            if not any(isinstance(item, allowed_type) for allowed_type in self.allowed_types):
                raise TypeError(f"Item {item} is not an instance of any allowed types: {self.allowed_types}")
        return item

    def _raise_type_error(self):
        raise TypeError(f"Items must be instances of: {', '.join(self.allowed_types)}")

    async def put(self, item):
        self._check_item(item)
        await super().put(item)

    def put_nowait(self, item):
        self._check_item(item)
        super().put_nowait(item)


class Thinking(BaseModel):
    xml: str = Field(description='xml string')
    regex: str = Field(description='regex string to enforce any value', default=None)



class ModelParser(BaseModel):
    model_id: str = Field(description='id of the model from the yml file')
    model_name: Optional[str] = Field(description='name of the model', default=None)
    gpus: Optional[List[float]] = Field(description='VRam usage for each GPU', default=None)
    cache_size: Optional[int] = Field(default=None, description='The context length for cache text in int. If None, will be set to the model context length')
    cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default=None, description='the quantization to use for cache, will use Q4 if not specified')
    max_seq_len: Optional[int] = Field(description="max sequence length", default=None)
    backend: Optional[Union[Literal["exllama", "llama_cpp", "embedding"], None]] = Field(description="model engine backend", default=None)
    tensor_parallel: Optional[bool] = Field(description="tensor parallel mode", default=False)

    # speculative decoding
    draft_model_id: Optional[str] = Field(description='id of the draft model', default=None)
    draft_model_name: Optional[str] = Field(description='name of the draft model', default=None)
    draft_gpus: Optional[List[float]] = Field(description='VRam usage for each GPU', default=None)
    draft_cache_size: Optional[int] = Field(description='The context length for cache text in int. If None, will be set to the model context length', default=None)
    draft_cache_quant: Optional[Literal["FP16", "Q4", "Q6", "Q8"]] = Field(default=None, description='the quantization to use for cache, will use Q4 if not specified')
    # backend is assumed to be the same as main model



    # dont allow non recognizable option
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())  # disable protected_namespaces due to it field use model_ in the name

    @validator('model_name', pre=True, always=True)
    def set_model_name(cls, v, values):
        if v is None and 'model_id' in values:
            return values['model_id'].split('/')[-1]
        return v

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

        return gpus

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
        #model_name = input_dict.get('model_name')
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
        return cls(model_id=model_id, gpus=gpus, cache_size=cache_size, backend=backend, cache_quant=cache_quant,
                   max_seq_len=max_seq_len,
                   tensor_parallel=tensor_parallel,
                   draft_model_id=draft_model_id, draft_model_name=draft_model_name,
                   draft_gpus=draft_gpus, draft_cache_size=draft_cache_size, draft_cache_quant=draft_cache_quant)

    def to_arg_string(self) -> str:
        """
        Generate a command-line argument string based on the instance's attributes.

        Returns:
            str: A string representation of the command-line arguments.
        """
        args = [f"model_id={self.model_id}"]

        if self.model_name != self.model_id.split('/')[-1]:
            args.append(f"model_name={self.model_name}")

        if self.gpus is not None:
            args.append(f"gpus={','.join(str(vram) for vram in self.gpus)}")

        if self.max_seq_len is not None:
            args.append(f"max_seq_len={self.max_seq_len}")

        if self.cache_size is not None:
            args.append(f"cache_size={self.cache_size}")

        if self.cache_quant is not None:
            args.append(f"cache_quant={self.cache_quant}")

        if self.backend != "exllama":  # Only include if it's not the default value
            args.append(f"backend={self.backend}")

        if self.tensor_parallel:
            args.append(f"tp={self.tensor_parallel}")

        # Add draft model parameters
        if self.draft_model_id is not None:
            args.append(f"draft_model_id={self.draft_model_id}")

        if self.draft_model_name is not None and self.draft_model_name != self.draft_model_id.split('/')[-1]:
            args.append(f"draft_model_name={self.draft_model_name}")

        if self.draft_gpus is not None:
            args.append(f"draft_gpus={','.join(str(vram) for vram in self.draft_gpus)}")

        if self.draft_cache_size is not None:
            args.append(f"draft_cache_size={self.draft_cache_size}")

        if self.draft_cache_quant is not None:
            args.append(f"draft_cache_quant={self.draft_cache_quant}")

        return " ".join(args)

    def get_visible_gpu_indices(self) -> str:
        """
        Generate a string of GPU indices based on allocated GPUs.
        If no GPUs are specified, return all available GPU indices.

        Returns:
            str: A comma-separated string of GPU indices with allocated VRAM,
                 or all available GPU indices if none are specified.
        """
        if self.gpus is None:
            import torch
            return ','.join(str(i) for i in range(torch.cuda.device_count()))

        if all(vram == 0 for vram in self.gpus):
            return ""  # No GPUs allocated

        visible_devices = [str(i) for i, vram in enumerate(self.gpus) if vram > 0]
        return ','.join(visible_devices)


class ModelDownloadSpec(BaseModel):
    """ dataclass for model download"""
    model_name: str
    quant: Optional[float] = None
    backend: Literal["exllama", "llama_cpp", "embedding"] = "exllama"

    # disable protected_namespaces due to it field use model_ in the name
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
