from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator, HttpUrl
from typing import Optional, Literal, List, Dict, Union, Any, Type, TypeVar
from dataclasses import dataclass
import asyncio
import weakref


class GenerationStats(BaseModel):
    input_tokens_count: int = Field(description='input tokens count', default=0)
    output_tokens_count: int = Field(description='output tokens count', default=0)
    time_to_first_token: float = Field(description='time to first token', default=0)
    time_generate: float = Field(description='time to generate tokens', default=0)
    cached_pages: int = Field(description='number of cached pages', default=0)
    cached_tokens: int = Field(description='number of cached tokens', default=0)

    @property
    def total_tokens_count(self) -> int:
        return self.input_tokens_count + self.output_tokens_count

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


T = TypeVar('T', bound='GenQueueDynamic')

class GenQueueDynamic:
    """
    A manager class that behaves like a GenQueue by default but allows swapping the underlying queue.
    """
    def __init__(self,
        maxsize=0,
        allowed_types: Optional[List[BaseModel]] = None,
        include_GenStats: bool =True,
        include_GenEnd: bool =True,
        existing_queue: Optional[GenQueue] = None
    ):
        # Initialize with a GenQueue as the active queue
        self._include_GenStats: bool = include_GenStats
        self._include_GenEnd: bool = include_GenEnd

        if not existing_queue:
            self._active_queue = GenQueue(maxsize=maxsize, allowed_types=allowed_types)
        else:
            self._active_queue = existing_queue

    @property
    def include_GenStats(self) -> bool:
        return self._include_GenStats

    @property
    def include_GenEnd(self) -> bool:
        return self._include_GenEnd

    def swap(self, new_queue: GenQueue | T):
        """
        Swap the active queue to the provided GenQueue instance.
        """
        if isinstance(new_queue, GenQueueDynamic):
            self._active_queue = new_queue._active_queue
        elif not isinstance(new_queue, GenQueue):
            raise TypeError(f"new_queue must be an instance of GenQueue, it was of type: {type(new_queue)}")
        self._active_queue = new_queue

    async def put(self, item):
        """
        Put an item into the active queue.
        """
        await self._active_queue.put(item)

    async def get(self):
        """
        Get an item from the active queue.
        """
        return await self._active_queue.get()

    def put_nowait(self, item):
        """
        Put an item into the active queue without waiting.
        """
        self._active_queue.put_nowait(item)

    def get_nowait(self):
        """
        Get an item from the active queue without waiting.
        """
        return self._active_queue.get_nowait()

    def empty(self):
        """
        Check if the active queue is empty.
        """
        return self._active_queue.empty()

    def qsize(self):
        """
        Get the size of the active queue.
        """
        return self._active_queue.qsize()

    def task_done(self):
        """
        Indicate that a formerly enqueued task is complete.
        """
        self._active_queue.task_done()

    def join(self):
        """
        Block until all items in the active queue have been processed.
        """
        return self._active_queue.join()



@dataclass
class QueueContext:
    """ helper class for short live handling of multiple Queue"""
    # gen_queue: 'weakref.ReferenceType[asyncio.Queue]'
    gen_queue: Union[GenQueue, GenQueueDynamic]
    include_GenStats: bool
    include_GenEnd: bool

    @classmethod
    def create(cls, gen_queue: GenQueue | GenQueueDynamic, include_GenStats=True, include_GenEnd=True) -> 'QueueContext':
        """
        Factory method to create a QueueContext instance.
        Args:
            gen_queue: An instance of QueueManager.
            include_GenStats: Whether to include GenerationStats in the queue.
            include_GenEnd: Whether to include GenEnd in the queue.
        """
        if isinstance(gen_queue, GenQueue):
            # Wrap the GenQueue in a GenQueueDynamic
            gen_queue = GenQueueDynamic(existing_queue=gen_queue)
        return cls(
            gen_queue=gen_queue,
            include_GenStats=include_GenStats,
            include_GenEnd=include_GenEnd
        )


    def get_queue(self) -> GenQueue | None:
        return self.gen_queue

