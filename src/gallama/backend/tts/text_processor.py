from typing import AsyncIterator, List, Callable, Optional, Literal
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
                 segment_size: int = 4,
                 char_limit: int = 50,
                 min_segment_length: int = 5,
                 max_queue_size: int = 100,
                 buffer_size: int = 1000,
                 custom_segmenter: Optional[Callable[[str], List[str]]] = None,
                 quick_start: bool = False,
                 quick_start_min_words: int = 5,
                 quick_start_max_words: int = 15
                 ):
        """
        Initialize the pipeline with configurable parameters.
        """
        self.segmentation_method = segmentation_method
        self.segment_size = segment_size
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

    def _find_quick_start_break(self, text: str) -> int:
        """
        Find an appropriate break point for the quick start first segment.
        Returns the index where the first segment should end.
        """
        words = text.split()
        if len(words) < self.quick_start_min_words:
            return -1

        # First try to find a natural break point within our word limits
        current_text = ' '.join(words[:self.quick_start_max_words])
        for i in range(len(current_text)):
            if current_text[i] in self.PUNCTUATION_MARKS:
                word_count = len(current_text[:i + 1].split())
                if word_count >= self.quick_start_min_words:
                    return i + 1

        # If no punctuation found, break at max words
        if len(words) > self.quick_start_max_words:
            return len(' '.join(words[:self.quick_start_max_words]))

        return len(current_text)

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
        self.text_buffer += text

        if self.quick_start and not self.first_segment_processed:
            break_point = self._find_quick_start_break(self.text_buffer)
            if break_point > 0:
                complete_text = self.text_buffer[:break_point]
                self.text_buffer = self.text_buffer[break_point:]
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
        current_segment = []
        current_length = 0

        # Split into words first to avoid splitting within words
        words = text.split()

        for i, word in enumerate(words):
            word_length = len(word)

            # If adding this word would exceed the limit
            if current_length + word_length + 1 > self.char_limit:
                # If current segment is empty, force add the word
                if not current_segment:
                    current_segment.append(word)
                else:
                    # Complete current segment
                    segments.append(' '.join(current_segment))
                    current_segment = [word]
                    current_length = word_length
                    continue

            current_segment.append(word)
            current_length += word_length + 1  # +1 for space

            # Check if current word ends with sentence-ending punctuation
            if any(word.endswith(mark) for mark in {'.', '!', '?', '。', '！', '？'}):
                # Complete current segment at sentence boundary if it's long enough
                if current_length >= self.min_segment_length:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                    current_length = 0

        # Add remaining text if any
        if current_segment and len(' '.join(current_segment)) >= self.min_segment_length:
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
            return self._group_sentences(sentences, self.segment_size)

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
        text_iterator: AsyncIterator[str],
        end_stream: bool = True
    ) -> None:
        """
        Process a stream of text asynchronously, segmenting and adding to the queue.

        Args:
            text_iterator: AsyncIterator yielding text strings
            end_stream: Whether this is the end of the text stream
        """
        try:
            async for text in text_iterator:
                if self.stop_event.is_set():
                    break

                # Buffer text and get complete sentences
                complete_text = self._buffer_text(text)
                if complete_text:
                    segments = self.segment_text(complete_text)
                    for segment in segments:
                        await self.processing_queue.put(segment)

            # Handle end of stream if specified
            if end_stream and self.text_buffer:
                # Process any remaining text in buffer
                segments = self.segment_text(self.text_buffer)
                for segment in segments:
                    await self.processing_queue.put(segment)
                self.text_buffer = ""
                # Signal end of processing
                await self.processing_queue.put(None)

        finally:
            if end_stream:
                await self.processing_queue.put(None)

    async def get_next_segment(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Get the next text segment from the queue.
        Returns None if queue is empty and processing is finished.

        Args:
            timeout: How long to wait for next segment (seconds)

        Returns:
            Optional[str]: The next segment or None if timeout/finished
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.processing_queue.get(), timeout)
            return await self.processing_queue.get()
        except asyncio.TimeoutError:
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
        self.stop_event.clear()