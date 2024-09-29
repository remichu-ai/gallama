import asyncio
from queue import Queue
from typing import Optional, AsyncIterator
from transformers import TextIteratorStreamer, AutoTokenizer
import threading


class AsyncTextIteratorStreamer(TextIteratorStreamer):
    """
    Async version of TextIteratorStreamer for use with FastAPI and other async frameworks.

    This streamer inherits from TextIteratorStreamer and adapts it for both sync and async use.
    It maintains two queues: a synchronous one for compatibility with the parent class,
    and an asynchronous one for use in async contexts.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.
    """

    def __init__(
            self,
            tokenizer: "AutoTokenizer",
            skip_prompt: bool = False,
            timeout: Optional[float] = None,
            **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        # Keep the original queue for synchronous operations
        self.text_queue = Queue()
        # Add an async queue for asynchronous operations
        self.async_text_queue = asyncio.Queue()
        # Add an event to signal when we're done
        self.done = threading.Event()

    def put(self, value):
        """
        Overrides the put method to handle both sync and async queues.
        """
        super().put(value)

        # Add to the async queue in a thread-safe manner
        asyncio.run_coroutine_threadsafe(self.async_text_queue.put(value), asyncio.get_event_loop())

    def end(self):
        """
        Overrides the end method to handle both sync and async queues.
        """
        super().end()

        # Signal end in the async queue
        asyncio.run_coroutine_threadsafe(self.async_text_queue.put(self.stop_signal), asyncio.get_event_loop())
        # Set the done event
        self.done.set()

    async def __aiter__(self) -> AsyncIterator[str]:
        """
        Async iterator that yields generated text as it becomes available.
        """
        while not self.done.is_set() or not self.async_text_queue.empty():
            try:
                value = await asyncio.wait_for(self.async_text_queue.get(), timeout=self.timeout)
                if value == self.stop_signal:
                    break
                yield value
            except asyncio.TimeoutError:
                # Handle timeout if needed
                continue

    async def iter_text(self):
        """
        Async generator for use with FastAPI's StreamingResponse.
        """
        async for text in self:
            yield text