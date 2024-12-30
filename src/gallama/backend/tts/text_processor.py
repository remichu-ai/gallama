from typing import AsyncIterator, List, Callable, Optional, Literal
from ...logger.logger import logger
from ...data_classes import TTSEvent
import asyncio



class TextToTextSegment:
    """
    A pipeline for processing continuous text streams into audio segments with buffering.
    Handles text segmentation, queuing, and processing for text-to-audio generation.
    """

    PUNCTUATION_MARKS = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    SEGMENTATION_METHODS = Literal["sentence_group", "char_count", "period", "all_punctuation"]

    def __init__(self,
                 segmentation_method: SEGMENTATION_METHODS = "sentence_group",
                 initial_segment_size: int = 1,
                 max_segment_size: int = 4,
                 segment_size_increase_interval: int = 3,  # Number of segments before increasing
                 char_limit: int = 50,
                 min_segment_length: int = 5,
                 max_queue_size: int = 1000,
                 buffer_size: int = 500,
                 custom_segmenter: Optional[Callable[[str], List[str]]] = None,
                 quick_start: bool = False,
                 quick_start_min_words: int = 5,
                 quick_start_max_words: int = 15
                 ):
        """
        Initialize the pipeline with configurable parameters.
        Added parameters for dynamic segment size adjustment.
        """
        self.segmentation_method = segmentation_method
        self.initial_segment_size = initial_segment_size
        self.current_segment_size = initial_segment_size
        self.max_segment_size = max_segment_size
        self.segment_size_increase_interval = segment_size_increase_interval
        self.segments_processed = 0
        self.char_limit = char_limit
        self.min_segment_length = min_segment_length
        self.max_buffer_size = buffer_size
        self.processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self.custom_segmenter = custom_segmenter
        self.stop_event = asyncio.Event()
        self.text_buffer = ""

        # Quick start parameters
        self.quick_start = quick_start
        self.quick_start_min_words = quick_start_min_words
        self.quick_start_max_words = quick_start_max_words
        self.first_segment_processed = False

    def _maybe_increase_segment_size(self):
        """
        Increase segment size if appropriate conditions are met.
        """
        self.segments_processed += 1
        if (self.segments_processed % self.segment_size_increase_interval == 0 and
            self.current_segment_size < self.max_segment_size):
            self.current_segment_size += 1
            logger.info(f"Increased segment size to: {self.current_segment_size}")

    def _find_quick_start_break(self, text: str) -> int:
        # First check if we have complete words by looking at word boundaries
        words = text.split()
        if len(words) < self.quick_start_min_words:
            return -1

        # Find actual word boundaries in the original text
        word_boundaries = []
        start = 0
        in_word = False

        # Find all word boundaries
        for i, char in enumerate(text):
            if char.isspace():
                if in_word:
                    word_boundaries.append((start, i))
                    in_word = False
            else:
                if not in_word:
                    start = i
                    in_word = True

        # Add last word if it's complete (ends with space or punctuation)
        if in_word and (len(text) == len(text.rstrip()) or text[-1] in self.PUNCTUATION_MARKS):
            word_boundaries.append((start, len(text)))

        # Not enough complete words yet
        if len(word_boundaries) < self.quick_start_min_words:
            return -1

        # Try to find punctuation break at word boundary
        current_word_count = 0
        for start, end in word_boundaries:
            current_word_count += 1
            if current_word_count > self.quick_start_max_words:
                # Return the end of previous word boundary
                return word_boundaries[current_word_count - 2][1]

            if text[end - 1] in self.PUNCTUATION_MARKS and current_word_count >= self.quick_start_min_words:
                return end

        # If we have enough words but no punctuation, break at max words
        if len(word_boundaries) > self.quick_start_max_words:
            return word_boundaries[self.quick_start_max_words - 1][1]

        # Use all complete words if between min and max
        if len(word_boundaries) >= self.quick_start_min_words:
            return word_boundaries[-1][1]

        return -1

    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text ends with a sentence-ending punctuation mark."""
        return any(text.strip().endswith(mark) for mark in {'.', '!', '?', '。', '！', '？'})

    def _find_last_sentence_break(self, text: str) -> int:
        """Find the last complete sentence break in the text."""
        for i in range(len(text) - 1, -1, -1):
            if text[i] in {'.', '!', '?', '。', '！', '？'}:
                # Check if it's not a decimal point
                if text[i] == '.' and i > 0 and i < len(text) - 1:
                    if text[i - 1].isdigit() and text[i + 1].isdigit():
                        continue
                return i + 1
        return -1

    def _buffer_text(self, text: str) -> str:
        """
        Buffer incomplete text and return complete sentences.
        Now handles quick start optimization for first segment.
        """
        # Add space if the buffer isn't empty and doesn't end with a space
        # and the new text doesn't start with a space
        if (self.text_buffer and
                not self.text_buffer.endswith(' ') and
                not text.startswith(' ')):
            if any(self.text_buffer.rstrip().endswith(p) for p in self.PUNCTUATION_MARKS | {' '}):
                self.text_buffer += ' '
        self.text_buffer += text

        if self.quick_start and not self.first_segment_processed:
            break_point = self._find_quick_start_break(self.text_buffer)
            if break_point > 0:
                complete_text = self.text_buffer[:break_point]
                self.text_buffer = self.text_buffer[break_point:].lstrip()
                self.first_segment_processed = True
                return complete_text
            elif len(self.text_buffer) > self.max_buffer_size:
                complete_text = self.text_buffer
                self.text_buffer = ""
                self.first_segment_processed = True
                return complete_text
            return ""

        # Regular buffering logic for subsequent segments
        break_point = self._find_last_sentence_break(self.text_buffer)

        if break_point == -1:
            if len(self.text_buffer) > self.max_buffer_size:
                complete_text = self.text_buffer
                self.text_buffer = ""
                return complete_text
            return ""

        complete_text = self.text_buffer[:break_point]
        self.text_buffer = self.text_buffer[break_point:]
        return complete_text

    def _split_by_punctuation(self, text: str) -> List[str]:
        """Split text at punctuation marks while preserving them."""
        segments = []
        current_segment = []

        for i, char in enumerate(text):
            current_segment.append(char)

            if char in self.PUNCTUATION_MARKS:
                # Don't split numbers with decimal points
                if (char == '.' and i > 0 and i < len(text) - 1
                        and text[i - 1].isdigit() and text[i + 1].isdigit()):
                    continue

                segments.append(''.join(current_segment))
                current_segment = []

        if current_segment:
            segments.append(''.join(current_segment))

        return [seg for seg in segments if len(seg.strip()) >= self.min_segment_length]

    def _group_sentences(self, sentences: List[str], group_size: int) -> List[str]:
        """Group sentences into chunks of specified size."""
        grouped = []
        current_group = []

        for sentence in sentences:
            current_group.append(sentence)
            if len(current_group) >= group_size:
                grouped.append(''.join(current_group))
                current_group = []

        if current_group:
            if len(grouped) > 0 and len(''.join(current_group)) < self.min_segment_length:
                grouped[-1] += ''.join(current_group)
            else:
                grouped.append(''.join(current_group))

        return grouped

    def _segment_by_char_count(self, text: str) -> List[str]:
        """
        Split text into segments based on character count while trying to maintain sentence integrity.

        Args:
            text: Input text to segment

        Returns:
            List of text segments, each approximately char_limit characters long
        """
        segments = []
        words = text.split()
        current_segment = []
        current_length = 0

        for i, word in enumerate(words):
            word_length = len(word)

            # Check if adding this word would exceed the limit
            if current_length + word_length + (1 if current_segment else 0) > self.char_limit:
                if current_segment:
                    # Properly join with spaces
                    segments.append(' '.join(current_segment))
                    current_segment = [word]
                    current_length = word_length
                else:
                    # Handle case where a single word exceeds limit
                    current_segment = [word]
                    current_length = word_length
            else:
                current_segment.append(word)
                current_length += word_length + (1 if current_segment else 0)

        # Handle remaining text
        if current_segment:
            segments.append(' '.join(current_segment))

        return segments


    def segment_text(self, text: str) -> List[str]:
        """
        Segment text using the specified method.
        Quick start optimization only applies to first segment through _buffer_text.
        """
        if not text or len(text.strip()) == 0:
            return []

        text = text.strip()

        if self.custom_segmenter is not None:
            return self.custom_segmenter(text)

        if self.segmentation_method == "sentence_group":
            sentences = self._split_by_punctuation(text)
            segments = self._group_sentences(sentences, self.current_segment_size)
            self._maybe_increase_segment_size()
            return segments

        elif self.segmentation_method == "char_count":
            return self._segment_by_char_count(text)

        elif self.segmentation_method == "period":
            return [s.strip() for s in text.split('.') if len(s.strip()) >= self.min_segment_length]

        elif self.segmentation_method == "all_punctuation":
            return self._split_by_punctuation(text)

        else:
            raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")

    def get_buffer_contents(self) -> str:
        """Get current contents of the text buffer."""
        return self.text_buffer

    def clear_buffer(self):
        """Clear the text buffer."""
        self.text_buffer = ""

    async def process_text_stream_async(
        self,
        input_queue: asyncio.Queue,
    ) -> None:
        """
        Process text from an input queue asynchronously, handling both TTSEvent and string inputs.

        Args:
            input_queue: AsyncQueue containing either TTSEvent objects or text strings
            end_stream: Whether to process remaining buffer when stream ends
        """
        try:
            logger.info("Starting text stream processing")

            while True:
                if self.stop_event.is_set():
                    break

                # Get next item from queue
                item = await input_queue.get()
                # Handle string input
                if isinstance(item, str):
                    complete_text = self._buffer_text(item)
                    if complete_text:
                        segments = self.segment_text(complete_text)
                        for segment in segments:
                            logger.debug(f"segment: {segment}")
                            await self.processing_queue.put(segment)

                # Handle TTSEvent
                elif isinstance(item, TTSEvent):
                    if item.type == "text_end":
                        # Process remaining buffer if any
                        if self.text_buffer:
                            segments = self.segment_text(self.text_buffer)
                            for segment in segments:
                                await self.processing_queue.put(segment)
                            self.text_buffer = ""
                        break
                    elif item.type == "text_start":
                        pass

            logger.info("End of stream processing complete")
        finally:
            pass

    async def get_next_segment(
        self,
        timeout: Optional[float] = None,
        raise_timeout: bool = False
    ) -> Optional[str]:
        """
        Get the next text segment from the queue.
        Returns None if queue is empty and processing is finished.

        Args:
            timeout: How long to wait for next segment (seconds)
            raise_timeout: If True, raises TimeoutError instead of returning None

        Returns:
            Optional[str]: The next segment or None if timeout/finished

        Raises:
            asyncio.TimeoutError: If timeout occurs and raise_timeout is True
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.processing_queue.get(), timeout)
            return await self.processing_queue.get()
        except asyncio.TimeoutError:
            if raise_timeout:
                raise
            return None
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            return None
        finally:
            # Only call task_done() if we actually got an item
            if not self.processing_queue.empty():
                self.processing_queue.task_done()

    async def stop_processing(self):
        """Stop the processing pipeline."""
        self.stop_event.set()
        self.clear_buffer()

    async def clear_queue(self):
        """Clear the processing queue."""
        try:
            while True:
                self.processing_queue.get_nowait()
                self.processing_queue.task_done()
        except asyncio.QueueEmpty:
            pass

    async def reset(self):
        """Reset the pipeline state."""
        self.clear_buffer()
        await self.clear_queue()
        self.first_segment_processed = False
        self.segments_processed = 0
        self.current_segment_size = self.initial_segment_size
        self.stop_event.clear()