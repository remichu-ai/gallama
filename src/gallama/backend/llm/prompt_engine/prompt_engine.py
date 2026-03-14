from .pe_custom import PromptEngineCustom
from .pe_transformers import PromptEngineTransformers
from typing import Protocol, runtime_checkable
from typing import List, Dict, Union, Literal
from gallama.data_classes.data_class import (
    BaseMessage,
    ChatMLQuery,
    ToolCall,
    MultiModalTextContent,
    MultiModalImageContent,
    MultiModalImageHFContent,
    MultiModalAudioContent
)
from pydantic import BaseModel

@runtime_checkable
class _Impl(Protocol):
    def get_prompt(
        self,
        query: ChatMLQuery,
        pydantic_tool_dict: List[BaseModel] = None,
        answer_format_schema: bool = True,  # whether to add the instruction for tool calling answer schema
        prefix_prompt: str = "",
        leading_prompt: str = "",
        thinking_template: str = None,
        thinking_response: str = None,
        backend: Literal["exllama", "llama_cpp", "transformers", "embedding"] = "exllama",
        # skip model pseudo token and use exllama placeholder token # TODO - code refractoring
        pydantic_tool_code: str = None,  # the code representation of tool
    ) -> str: ...




class PromptEngine:
    def __init__(self, prompt_format: str | None = None, model_path: str = None):
        if prompt_format is None:
            self._impl = PromptEngineTransformers(prompt_format=prompt_format, model_path=model_path)
        else:
            self._impl = PromptEngineCustom(prompt_format=prompt_format, model_path=model_path)

    def get_prompt(
        self,
        query: ChatMLQuery,
        pydantic_tool_dict: List[BaseModel] = None,
        answer_format_schema: bool = True,  # whether to add the instruction for tool calling answer schema
        prefix_prompt: str = "",
        leading_prompt: str = "",
        thinking_template: str = None,
        thinking_response: str = None,
        backend: Literal["exllama", "llama_cpp", "transformers", "embedding"] = "exllama",
        # skip model pseudo token and use exllama placeholder token # TODO - code refractoring
        pydantic_tool_code: str = None,  # the code representation of tool
    ) -> str:
        return self._impl.get_prompt(
            query=query,
            pydantic_tool_dict=pydantic_tool_dict,
            answer_format_schema=answer_format_schema,
            prefix_prompt=prefix_prompt,
            leading_prompt=leading_prompt,
            thinking_template=thinking_template,
            thinking_response=thinking_response,
            backend=backend,
            pydantic_tool_code=pydantic_tool_code,
        )

    @property
    def eos_token_list(self):
        return self._impl.eos_token_list

    @property
    def tag_definitions(self):
        return self._impl.tag_definitions

    @property
    def tag_dict(self):
        return self._impl.special_tag

    @property
    def is_thinking(self):
        return self._impl.is_thinking

    @property
    def vision_token(self) -> str | None:
        return self._impl.vision_token
