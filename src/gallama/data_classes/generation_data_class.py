from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator, HttpUrl
from typing import Optional, Literal, List, Dict, Union, Any, Type
from dataclasses import dataclass
import asyncio
import weakref


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
    def __init__(self, maxsize=0, allowed_types: List[BaseModel] = None):
        super().__init__(maxsize)
        if allowed_types:
            self.allowed_types = allowed_types
        else:
            self.allowed_types = [GenText, GenerationStats, GenStart, GenEnd]

    def _check_item(self, item):
        if self.allowed_types:
            if not any(isinstance(item, allowed_type) for allowed_type in self.allowed_types):
                raise TypeError(f"Item {item} is not an instance of any allowed types: {self.allowed_types}")
        return item

    async def put(self, item):
        self._check_item(item)
        await super().put(item)

    def put_nowait(self, item):
        self._check_item(item)
        super().put_nowait(item)




@dataclass
class QueueContext:
    """ helper class for short live handling of multiple Queue"""
    gen_queue: 'weakref.ReferenceType[asyncio.Queue]'
    include_GenStats: bool
    include_GenEnd: bool

    @classmethod
    def create(cls, gen_queue: GenQueue, include_GenStats=True, include_GenEnd=True) -> 'QueueContext':
        return cls(
            gen_queue=weakref.ref(gen_queue),
            include_GenStats=include_GenStats,
            include_GenEnd=include_GenEnd
        )

    def get_queue(self) -> GenQueue | None:
        return self.gen_queue()