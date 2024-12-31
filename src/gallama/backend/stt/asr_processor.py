from .hypothesis import HypothesisBuffer
from .base import ASRBase
import numpy as np
from typing import Union, BinaryIO
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import soundfile as sf
from ...data_classes import TranscriptionResponse
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
        buffer_trimming=("segment", 15),
        min_context_needed = 5.0,   # require 5 second for a content to be process -> higher number better accuracy
        debug_audio_dir=None  # New parameter for debug audio directory
    ):
        """
        Initializes the ASR processor.

        Args:
            asr: ASR backend object for transcribing audio.
            tokenizer: Sentence tokenizer for the target language. Required if trimming mode is "sentence".
            buffer_trimming: Tuple (mode, duration).
                             - mode: "sentence" or "segment" (trimming method).
                             - duration: Max buffer length (in seconds) before trimming occurs.
            debug_audio_dir: Directory to store debug audio files
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.debug_audio_dir = debug_audio_dir
        self.min_context_needed = min_context_needed

        # Create debug directory if it doesn't exist
        if debug_audio_dir and not os.path.exists(debug_audio_dir):
            os.makedirs(debug_audio_dir)

        # Trimming configuration
        self.trimming_mode, self.trimming_duration = buffer_trimming

        # initial state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.audio_time_offset = 0
        self.transcript_buffer.last_committed_time = self.audio_time_offset
        self.committed_transcriptions = []
        self.buffer_count = 0  # Counter for debug files

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

    def save_debug_audio(self, audio_data: np.ndarray, suffix: str = ""):
        """
        Saves the current audio buffer to a WAV file for debugging.

        Args:
            audio_data: Audio data to save
            suffix: Optional suffix to add to the filename
        """
        if self.debug_audio_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_buffer_{timestamp}_{self.buffer_count}{suffix}.wav"
            filepath = os.path.join(self.debug_audio_dir, filename)

            try:
                sf.write(filepath, audio_data, self.SAMPLING_RATE)
                logger.info(f"Saved debug audio to {filepath}")
            except Exception as e:
                logger.error(f"Error saving debug audio: {str(e)}")

            self.buffer_count += 1
        else:
            pass

    def process_audio(self, is_final: bool = False):
        """
        Processes the current audio buffer, transcribes it, and handles buffer trimming.

        Args:
            is_final (bool): Whether this is the final processing call (end of stream)

        Returns:
            A tuple (start_time, end_time, transcription) for newly committed transcriptions.
        """
        # Check if audio buffer is empty
        if len(self.audio_buffer) == 0:
            logger.debug("Audio buffer is empty, returning early")
            return None, None, ""

        # Save the audio buffer before processing
        self.save_debug_audio(self.audio_buffer, "_before")

        # Generate the prompt and context
        prompt, context = self.construct_prompt()

        current_duration = len(self.audio_buffer) / self.SAMPLING_RATE

        # Only check for minimum context if not final
        if current_duration < self.min_context_needed and not is_final:
            return None, None, ""  # Wait for more context

        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {context}")
        logger.debug(
            f"Transcribing {current_duration:.2f} seconds from offset {self.audio_time_offset:.2f}")

        try:
            # Perform transcription
            asr_results = self.asr.transcribe_to_segment(self.audio_buffer, init_prompt=prompt)

            # Process timestamped words
            timestamped_words = self.asr.segment_to_timestamped_words(asr_results)
            self.transcript_buffer.insert(timestamped_words, self.audio_time_offset)

            # Get transcriptions based on whether this is final processing or not
            if is_final:
                # For final processing, commit all remaining words
                confirmed_transcriptions = self.transcript_buffer.set_final()
                # Save final audio buffer
                self.save_debug_audio(self.audio_buffer, "_final")
            else:
                # For normal streaming, use regular flush
                confirmed_transcriptions = self.transcript_buffer.flush()

            # Add to committed transcriptions list
            self.committed_transcriptions.extend(confirmed_transcriptions)

            # Handle trimming based on confirmed transcriptions
            if confirmed_transcriptions and self.trimming_mode == "sentence" and not is_final:
                if current_duration > self.trimming_duration:
                    self.trim_to_last_completed_sentence()

            # Handle segment-based trimming or fallback
            if not is_final:  # Don't trim on final processing
                if self.trimming_mode == "segment":
                    if current_duration > self.trimming_duration:
                        # Save audio before trimming
                        self.save_debug_audio(self.audio_buffer, "_before_trim")
                        self.trim_to_last_completed_segment(asr_results)
                        # Save audio after trimming
                        self.save_debug_audio(self.audio_buffer, "_after_trim")

            logger.info(
                f"Audio buffer length after processing: {current_duration:.2f} seconds")
            return self.format_output(confirmed_transcriptions)

        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            return None, None, ""  # Return empty result on error


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
