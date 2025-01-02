from .hypothesis import HypothesisBuffer
from .base import ASRBase
import numpy as np
from typing import Union, BinaryIO, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import soundfile as sf
from ...data_classes import TranscriptionResponse
from ...data_classes.realtime_client_proto import TurnDetectionConfig
from .audio_buffer import AudioBufferWithTiming
from .vad import VADProcessor
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
        min_context_needed=5.0,  # require 5 second for a content to be process -> higher number better accuracy
        debug_audio_dir=None,  # New parameter for debug audio directory
        vad_enable: bool = True,
        vad_config: Optional[TurnDetectionConfig] = None,
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

        # Initialize state with AudioBufferWithTiming
        self.audio_buffer = AudioBufferWithTiming(sample_rate=self.SAMPLING_RATE)
        self.transcript_buffer = HypothesisBuffer()
        self.transcript_buffer.last_committed_time = 0
        self.committed_transcriptions = []
        self.buffer_count = 0  # Counter for debug files

        self.vad_enable: bool = vad_enable
        self.vad_config = vad_config if vad_config else TurnDetectionConfig()
        self.vad = None
        if self.vad_enable:
            self.initialize_vad(self.vad_config)

        self.vad_speech_active = False  # Track if we're currently in a speech segment
        self.vad_start_sent = False  # Track if we've sent the start event for current segment
        self.vad_end_sent = False  # Track if we've sent the end event for current segment

    def initialize_vad(self, vad_config: TurnDetectionConfig):
        logger.info("Initializing VAD processor")
        self.vad = VADProcessor(vad_config)



    def reset_state(self, initial_offset=0):
        """Resets the internal state of the processor."""
        self.audio_buffer.reset()
        self.transcript_buffer = HypothesisBuffer()
        self.transcript_buffer.last_committed_time = initial_offset * 1000
        self.committed_transcriptions = []
        # Reset VAD state tracking
        self.vad_speech_active = False
        self.vad_start_sent = False
        self.vad_end_sent = False
        if self.vad:
            self.vad.reset()


    def add_audio_chunk(self, audio_chunk):
        """
        Adds an audio chunk to the buffer.

        Args:
            audio_chunk: Array of audio samples to append to the buffer.
        """
        self.audio_buffer.add_chunk(audio_chunk)

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

    def process_audio(self, is_final: bool = False):
        """
        Processes the current audio buffer, transcribes it, and handles buffer trimming.
        Now includes VAD processing with proper event state tracking.

        Args:
            is_final (bool): Whether this is the final processing call (end of stream)

        Returns:
            Tuple of (start_time, end_time, transcription, vad_events)
            where vad_events is a list of dictionaries containing VAD events
        """
        # Check if audio buffer is empty
        if len(self.audio_buffer) == 0:
            logger.debug("Audio buffer is empty, returning early")
            return None, None, "", []

        # Mark the start of processing
        self.audio_buffer.mark_processing_start()

        # Get unprocessed audio
        audio_to_process = self.audio_buffer.get_unprocessed_audio()
        if len(audio_to_process) == 0:
            logger.debug("No new audio to process")
            return None, None, "", []

        # Save the audio buffer before processing if in debug mode
        if self.debug_audio_dir:
            self.save_debug_audio(audio_to_process, "_before")

        current_offset = self.audio_buffer.last_processed_sample
        current_time_ms = self.audio_buffer.get_time_ms(current_offset)

        # Initialize VAD events list
        vad_events = []

        # Process through VAD if enabled
        if self.vad_enable and self.vad:
            vad_result = self.vad.process_chunk(self.audio_buffer, current_offset)
            logger.debug(f"VAD result: {vad_result}")

            if vad_result['speech_detected'] and not self.vad_speech_active and not self.vad_start_sent:
                self.vad_speech_active = True
                self.vad_start_sent = True
                self.vad_end_sent = False
                vad_events.append({
                    'type': 'start',
                    'timestamp_ms': int(vad_result.get('start_time', 0.0)),
                    'confidence': vad_result.get('confidence', 0.0)
                })
                logger.info(f"Speech start event queued at {vad_result.get('start_time', 0.0)}ms")

            if (vad_result['speech_ended'] or (is_final and self.vad_speech_active)) and not self.vad_end_sent:
                self.vad_speech_active = False
                self.vad_end_sent = True
                self.vad_start_sent = False

                # Use last_speech_ms from VADProcessor for the end timestamp
                end_timestamp = self.vad.last_speech_ms if self.vad else self.audio_buffer.get_time_ms(current_offset)

                vad_events.append({
                    'type': 'end',
                    'timestamp_ms': int(end_timestamp),
                    'confidence': vad_result.get('confidence', 0.0)
                })
                logger.info(f"Speech end event queued at {end_timestamp}ms")

            # If VAD is enabled and no speech is detected, skip processing
            if not vad_result['speech_detected'] and not is_final:
                self.audio_buffer.mark_processing_complete(False)
                return None, None, "", vad_events

        # Calculate current duration
        current_duration_ms = (len(audio_to_process) / self.SAMPLING_RATE) * 1000
        current_duration = current_duration_ms / 1000  # Convert to seconds

        # Check for minimum context if not final
        if current_duration < self.min_context_needed and not is_final:
            self.audio_buffer.is_processing = False
            return None, None, "", vad_events

        try:
            # Generate the prompt and context
            prompt, context = self.construct_prompt()

            # Perform transcription on the unprocessed portion
            asr_results = self.asr.transcribe_to_segment(audio_to_process, init_prompt=prompt)

            # Process timestamped words
            timestamped_words = self.asr.segment_to_timestamped_words(asr_results)
            buffer_start_time = ((self.audio_buffer.start_offset + self.audio_buffer.last_processed_sample)
                                 / self.SAMPLING_RATE)
            self.transcript_buffer.insert(timestamped_words, buffer_start_time)

            # Get transcriptions based on whether this is final processing or not
            if is_final:
                confirmed_transcriptions = self.transcript_buffer.set_final()
                if self.debug_audio_dir:
                    self.save_debug_audio(audio_to_process, "_final")
            else:
                confirmed_transcriptions = self.transcript_buffer.flush()

            # Add to committed transcriptions list
            self.committed_transcriptions.extend(confirmed_transcriptions)

            # Handle trimming based on confirmed transcriptions
            if not is_final:  # Don't trim on final processing
                if self.trimming_mode == "sentence" and confirmed_transcriptions:
                    if current_duration > self.trimming_duration:
                        self.trim_to_last_completed_sentence()
                elif self.trimming_mode == "segment":
                    if current_duration > self.trimming_duration:
                        if self.debug_audio_dir:
                            self.save_debug_audio(self.audio_buffer.buffer, "_before_trim")
                        self.trim_to_last_completed_segment(asr_results)
                        if self.debug_audio_dir:
                            self.save_debug_audio(self.audio_buffer.buffer, "_after_trim")

            # Mark processing complete and update the last processed position
            self.audio_buffer.mark_processing_complete(is_final)

            logger.info(f"Audio buffer length after processing: {current_duration:.2f} seconds")

            # Format and return the output along with VAD events
            start_time, end_time, transcription = self.format_output(confirmed_transcriptions)
            return start_time, end_time, transcription, vad_events

        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            self.audio_buffer.is_processing = False
            return None, None, "", vad_events


    def trim_buffer_at(self, timestamp):
        """
        Trims the audio buffer and transcription buffer at a specified timestamp.

        Args:
            timestamp: The time (in seconds) to trim the buffers up to.
        """
        self.transcript_buffer.pop_committed(timestamp)
        cut_samples = int((timestamp - (self.audio_buffer.start_offset / self.SAMPLING_RATE)) * self.SAMPLING_RATE)
        self.audio_buffer.clear_until(cut_samples)

    # Rest of the methods remain the same as they don't directly interact with the buffer
    def transcribe(self, audio: Union[str, BinaryIO, np.ndarray], init_prompt: str = "",
                   temperature: float = 0.0, language: str = None,
                   include_segments: bool = False) -> TranscriptionResponse:
        return self.asr.transcribe(
            audio,
            init_prompt=init_prompt,
            temperature=temperature,
            language=language,
            include_segments=include_segments
        )

    async def transcribe_async(self, audio: Union[str, BinaryIO, np.ndarray], init_prompt: str = "",
                               temperature: float = 0.0, language: str = None,
                               include_segments: bool = False) -> TranscriptionResponse:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(
                pool,
                self.transcribe,
                audio, init_prompt, temperature, language, include_segments
            )


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
