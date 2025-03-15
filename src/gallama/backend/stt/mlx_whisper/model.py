from ..base import ASRBase
from typing import Literal, List, Union, BinaryIO, Optional
import numpy as np

from ....logger import logger
from ....data_classes import ModelSpec, TranscriptionResponse, TimeStampedWord, LanguageType
import soundfile as sf


# faster whisper wont work. For now we import to reuse the data classes
# TODO create the data classes in gallama
try:
    from faster_whisper.transcribe import TranscriptionOptions, TranscriptionInfo, Segment
except ImportError:
    TranscriptionOptions, TranscriptionInfo, Segment = None, None, None, None, None


import mlx_whisper as mlx_model
import mlx.core as mx


class ASRMLXWhisper(ASRBase):
    """
    An implementation of ASRBase that uses the MLX library as backend.
    """


    def load_model(self, model_spec: ModelSpec):
        """
        For MLX, we simply store the model parameters for use with the MLX transcribe function.
        """
        self.model_id = model_spec.model_id  # path or HF repo for the MLX converted weights
        self.model_name = model_spec.model_name
        self.quant = model_spec.quant

        # mlx whisper doesnt have dedicated load method
        # but it will load when transcribe is ran
        mlx_model.transcribe(
            audio=np.random.randn(44100 * 2),
            path_or_hf_repo=model_spec.model_id,
        )

        # Use fp16 if quantization is "16.0"; otherwise, assume fp32.
        # self.fp16 = True if self.quant == "16.0" else False
        return mlx_model

    # @staticmethod
    # def binaryio_to_numpy(audio_io: BinaryIO) -> Tuple[np.ndarray, int]:
    #     """
    #     Convert a BinaryIO audio stream into a numpy array.
    #     Returns a tuple of (audio_data, sample_rate).
    #     """
    #     audio_io.seek(0)  # Ensure the file pointer is at the beginning
    #     audio_data, sample_rate = sf.read(audio_io)
    #     return audio_data, sample_rate

    def transcribe_to_segment(
        self,
        audio: Union[str, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: LanguageType = None,
        batch: bool = False,
        batch_size: int = 8,
    ) -> List[dict]:
        """
        Transcribe audio into a list of segments.
        Calls the MLX transcribe function with word timestamps enabled.
        """

        if not isinstance(audio, np.ndarray):
            audio_to_use, sample_rate = sf.read(audio)
        else:
            audio_to_use = audio

        response = self.model.transcribe(
            audio=audio_to_use,
            path_or_hf_repo=self.model_id,  # eventhough loaded, we still need to use this
            temperature=temperature,
            condition_on_previous_text=True,
            initial_prompt=init_prompt,
            word_timestamps=True,
            hallucination_silence_threshold=1,
            **self.transcribe_kargs  # any extra keyword arguments set via use_vad(), etc.
        )
        segments = response.get("segments", [])
        return segments

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language=None,
        include_segments: bool = False,
        batch: bool = False,
    ) -> TranscriptionResponse:
        """
        Transcribe audio to full text and return a TranscriptionResponse.
        The response includes the text, detected language, duration (from the last segment),
        and word-level timestamps (either from MLX if available, or approximated).
        """

        if not isinstance(audio, np.ndarray):
            audio_to_use, sample_rate = sf.read(audio)
        else:
            audio_to_use = audio


        response = self.model.transcribe(
            audio=audio_to_use,
            path_or_hf_repo=self.model_id,  # eventhough loaded, we still need to use this
            temperature=temperature,
            condition_on_previous_text=True,
            initial_prompt=init_prompt,
            word_timestamps=True,
            hallucination_silence_threshold=1,
            **self.transcribe_kargs  # any extra keyword arguments set via use_vad(), etc.
        )
        text = response.get("text", "")
        segments = response.get("segments", [])
        lang = response.get("language", None)
        duration = segments[-1]["end"] if segments else None
        # Process segments to extract or approximate word-level timestamps.
        words = self.segment_to_timestamped_words(segments)
        return TranscriptionResponse(
            text=text,
            segments=segments if include_segments else None,
            words=words,
            language=lang,
            duration=duration
        )

    def segment_to_timestamped_words(self, segments: List[dict]) -> List[TimeStampedWord]:
        """
        Converts a list of segments into word-level timestamps.
        If a segment includes a "words" key, it uses that data.
        Otherwise, it splits the segment text and assigns equal intervals.
        """
        output = []
        for segment in segments:
            # Skip segments that are likely silence.
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            if "words" in segment and segment["words"]:
                for word in segment["words"]:
                    output.append((word["start"], word["end"], word["word"]))
            else:
                # Fallback: split the segment text and assign approximate timestamps.
                words = segment["text"].split()
                if not words:
                    continue
                duration = segment["end"] - segment["start"]
                word_duration = duration / len(words)
                for i, w in enumerate(words):
                    start = segment["start"] + i * word_duration
                    end = start + word_duration
                    output.append((start, end, w))
        return output

    def segment_to_long_text(self, segments: List[Segment]) -> str:
        """
        Converts a list of segments into a full text.

        Args:
            segments (List[Segment]): Transcription segments with words and metadata.

        Returns:
            List[Tuple[float, float, str]]: List of (start, end, word) tuples.
        """

        output = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:    # silence segment
                continue
            for word in segment.words:
                output.append(word.word)

        return self.sep.join(output)

    def segments_end_ts(self, segments: List[dict]) -> List[float]:
        """
        Returns a list of ending timestamps from the provided segments.
        """
        return [segment.get("end", 0.0) for segment in segments]

    def use_vad(self):
        """
        Enable voice activity detection (VAD) settings.
        For example, here we set a no-speech threshold in the extra keyword arguments.
        """
        self.transcribe_kargs["no_speech_threshold"] = 0.6

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"