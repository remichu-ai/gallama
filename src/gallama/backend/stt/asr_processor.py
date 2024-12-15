from .hypothesis import HypothesisBuffer
from .base import ASRBase
import numpy as np
from typing import Union, BinaryIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel

# from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionOptions, TranscriptionInfo, Segment
from ...data_classes import TranscriptionResponse
from gallama.logger.logger import logger


import sys
from ...logger import logger


class ASRProcessor:
    """
    ASRProcessor processes audio buffers for real-time ASR (Automatic Speech Recognition).
    It manages an audio buffer, transcribes audio using an ASR backend, and trims processed portions
    to maintain efficiency. Supports trimming based on sentences or audio segments.
    """

    SAMPLING_RATE = 16000  # Standard audio sampling rate

    def __init__(
        self,
        asr: ASRBase,
        tokenizer=None,
        buffer_trimming=("segment", 15)
    ):
        """
        Initializes the ASR processor.

        Args:
            asr: ASR backend object for transcribing audio.
            tokenizer: Sentence tokenizer for the target language. Required if trimming mode is "sentence".
            buffer_trimming: Tuple (mode, duration).
                             - mode: "sentence" or "segment" (trimming method).
                             - duration: Max buffer length (in seconds) before trimming occurs.
        """
        self.asr = asr
        self.tokenizer = tokenizer

        # Trimming configuration
        self.trimming_mode, self.trimming_duration = buffer_trimming

        # initial state
        self.audio_buffer = np.array([], dtype=np.float32)              # Holds incoming audio chunks
        self.transcript_buffer = HypothesisBuffer()                             # Tracks ongoing hypotheses
        self.audio_time_offset = 0                                           # Current start time of the audio buffer
        self.transcript_buffer.last_committed_time = self.audio_time_offset
        self.committed_transcriptions = []                                      # List of confirmed transcriptions


    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: str = None,
        include_segments: bool = False      # whether to include segment data for openai api spec
    ) -> TranscriptionResponse:
        """
        transcribe audio to text
        """

        return self.asr.transcribe(
            audio,
            init_prompt=init_prompt,
            temperature=temperature,
            language=language,
            include_segments=include_segments
        )


    async def transcribe_async(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: str = None,
        include_segments: bool = False  # whether to include segment data for openai api spec
    ) -> TranscriptionResponse:

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as pool:
            transcribed_object = await loop.run_in_executor(
                pool,
                self.transcribe,
                audio, init_prompt, temperature, language, include_segments
            )

        return transcribed_object


    def reset_state(self, initial_offset=0):
        """
        Resets the internal state of the processor. Should be called on initialization or restart.

        Args:
            initial_offset: Initial time offset for the audio buffer (in seconds).
        """
        self.audio_buffer = np.array([], dtype=np.float32)              # Holds incoming audio chunks
        self.transcript_buffer = HypothesisBuffer()                             # Tracks ongoing hypotheses
        self.audio_time_offset = initial_offset                                 # Current start time of the audio buffer
        self.transcript_buffer.last_committed_time = self.audio_time_offset
        self.committed_transcriptions = []                                      # List of confirmed transcriptions


    def add_audio_chunk(self, audio_chunk):
        """
        Adds an audio chunk to the buffer.

        Args:
            audio_chunk: Array of audio samples to append to the buffer.
        """
        self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

    def construct_prompt(self):
        """
        Constructs a prompt from previously confirmed transcriptions for context during ASR.

        Returns:
            Tuple (prompt, context):
                - prompt: A 200-character string of previous context for the ASR model.
                - context: The remaining transcriptions within the current audio buffer for logging/debugging.
        """
        # Find the first committed transcription outside the current buffer
        start_index = max(0, len(self.committed_transcriptions) - 1)
        while start_index > 0 and self.committed_transcriptions[start_index - 1][1] > self.audio_time_offset:
            start_index -= 1

        past_context = self.committed_transcriptions[:start_index]
        past_texts = [text for _, _, text in past_context]

        # Build prompt (up to 200 characters)
        prompt, current_length = [], 0
        while past_texts and current_length < 200:
            last_text = past_texts.pop()
            current_length += len(last_text) + 1
            prompt.append(last_text)

        remaining_transcriptions = self.committed_transcriptions[start_index:]
        return self.asr.sep.join(reversed(prompt)), self.asr.sep.join(text for _, _, text in remaining_transcriptions)

    def process_audio(self):
        """
        Processes the current audio buffer, transcribes it, and handles buffer trimming.

        Returns:
            A tuple (start_time, end_time, transcription) for newly committed transcriptions.
        """
        # Generate the prompt and context
        prompt, context = self.construct_prompt()
        logger.info(f"PROMPT: {prompt}")
        logger.info(f"CONTEXT: {context}")
        logger.info(
            f"Transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:.2f} seconds from offset {self.audio_time_offset:.2f}")

        # Perform transcription
        asr_results = self.asr.transcribe_to_segment(self.audio_buffer, init_prompt=prompt)

        # Process timestamped words
        timestamped_words = self.asr.segment_to_timestamped_words(asr_results)
        self.transcript_buffer.insert(timestamped_words, self.audio_time_offset)

        # Commit transcriptions
        confirmed_transcriptions = self.transcript_buffer.flush()
        self.committed_transcriptions.extend(confirmed_transcriptions)

        # Handle trimming based on confirmed transcriptions
        if confirmed_transcriptions and self.trimming_mode == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.trimming_duration:
                self.trim_to_last_completed_sentence()

        # Handle segment-based trimming or fallback
        trim_duration = self.trimming_duration if self.trimming_mode == "segment" else 30
        if len(self.audio_buffer) / self.SAMPLING_RATE > trim_duration:
            self.trim_to_last_completed_segment(asr_results)

        logger.info(
            f"Audio buffer length after processing: {len(self.audio_buffer) / self.SAMPLING_RATE:.2f} seconds")
        return self.format_output(confirmed_transcriptions)

    def trim_to_last_completed_sentence(self):
        """
        Trims the audio buffer and committed transcriptions up to the end of the last completed sentence.
        """
        if not self.committed_transcriptions:
            return

        sentences = self.segment_transcriptions_into_sentences(self.committed_transcriptions)
        if len(sentences) < 2:
            return  # Not enough sentences to trim

        # Keep the last two sentences; trim up to the second-last one
        trim_timestamp = sentences[-2][1]
        self.trim_buffer_at(trim_timestamp)

    def trim_to_last_completed_segment(self, transcription_results):
        """
        Trims the audio buffer to the last completed audio segment.

        Args:
            transcription_results: Transcription results from the ASR system.
        """
        if not self.committed_transcriptions:
            return

        segment_end_times = self.asr.segments_end_ts(transcription_results)
        if len(segment_end_times) > 1:
            trim_timestamp = segment_end_times[-2] + self.audio_time_offset
            if trim_timestamp <= self.committed_transcriptions[-1][1]:
                self.trim_buffer_at(trim_timestamp)

    def trim_buffer_at(self, timestamp):
        """
        Trims the audio buffer and transcription buffer at a specified timestamp.

        Args:
            timestamp: The time (in seconds) to trim the buffers up to.
        """
        self.transcript_buffer.pop_committed(timestamp)
        cut_seconds = timestamp - self.audio_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.audio_time_offset = timestamp

    def segment_transcriptions_into_sentences(self, words):
        """
        Segments committed transcriptions into sentences using the tokenizer.

        Args:
            words: List of transcribed words with timestamps.

        Returns:
            List of tuples [(start_time, end_time, sentence), ...].
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for sentence segmentation.")

        complete_words = list(words)
        text = " ".join(word[2] for word in complete_words)
        sentences = self.tokenizer.split(text)

        segmented = []
        for sentence in sentences:
            start, end = None, None
            while complete_words:
                word_start, word_end, word = complete_words.pop(0)
                if start is None and sentence.startswith(word):
                    start = word_start
                if sentence.strip() == word:
                    end = word_end
                    segmented.append((start, end, sentence.strip()))
                    break
                sentence = sentence[len(word):].strip()
        return segmented

    def format_output(self, sentences):
        """
        Formats the confirmed sentences for output.

        Args:
            sentences: List of sentences [(start_time, end_time, sentence), ...].

        Returns:
            A tuple (start_time, end_time, combined_text) or (None, None, "") if empty.
        """
        if not sentences:
            return None, None, ""

        start_time = sentences[0][0]
        end_time = sentences[-1][1]
        combined_text = self.asr.sep.join(sentence[2] for sentence in sentences)
        return start_time, end_time, combined_text
