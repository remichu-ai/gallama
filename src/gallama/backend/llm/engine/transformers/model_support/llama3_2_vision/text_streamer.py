from transformers import TextIteratorStreamer
from queue import Queue
from typing import Optional


class CustomTextIteratorStreamer(TextIteratorStreamer):
    def __init__(
            self,
            tokenizer,  # Replace 'tokenizer' with 'processor'
            skip_prompt: bool = False,
            timeout: Optional[float] = None,
            **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.tokenizer = tokenizer  # Store the processor instead of tokenizer

    def put(self, value):
        """
        Receives tokens, decodes them, and puts them in the queue.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("CustomTextIteratorStreamer only supports batch size 1")

        # Use value[0] instead of value
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache
        self.token_cache.extend(value.tolist())

        # Use the processor to decode instead of tokenizer
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # Process the text similarly to the original TextStreamer
        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len: text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and signals the end of the stream."""
        if len(self.token_cache) > 0:
            # Use the processor to decode instead of tokenizer
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)