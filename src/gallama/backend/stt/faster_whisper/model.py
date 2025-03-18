from ..base import ASRBase
from ....logger import logger
from ....data_classes import ModelSpec, TranscriptionResponse, LanguageType, TimeStampedWord

import dataclasses
from typing import Literal, List, Union, BinaryIO, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from faster_whisper.transcribe import TranscriptionOptions, TranscriptionInfo, Segment
except ImportError:
    WhisperModel, BatchedInferencePipeline, TranscriptionOptions, TranscriptionInfo, Segment = None, None, None, None, None



class ASRFasterWhisper(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """

    class ModelWrapper:
        def __init__(self, model, model_batch):
            self.model = model
            self.model_batch = model_batch

        def transcribe(self, _type: Literal["single", "batch"] = "single", **kwargs):
            if _type == "single":
                logger.debug("using single transcribe")
                return self.model.transcribe(**kwargs)
            elif _type == "batch":
                logger.debug("using batch transcribe")
                return self.model_batch.transcribe(**kwargs)
            else:
                raise ValueError("type must be either 'single' or 'batch'")

        def detect_language(self, **kwargs):
            return self.model.detect_language(**kwargs)

    def load_model(
        self,
        model_spec: ModelSpec,
    ):

        # set seperator for faster whisper to ""
        self.sep = ""                       # seperator for faster whisper is ""
        self.compute_type = "float16"       # to store the quantization
        self.model_id = model_spec.model_id      # to store path to model
        self.model_name = model_spec.model_name
        self.device = "cuda" # if device == "gpu" else "cpu"


        if self.quant == "8.0":
            self.compute_type = "int8_float16"

        # load model
        model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            # download_root=cache_dir
        )

        model_batched = BatchedInferencePipeline(model)

        return self.ModelWrapper(model, model_batched)

    @staticmethod
    def _filter_lanague(allowed_lanague: List[str], all_language_probs: List[Tuple[str, float]]):
        """ find the most probable language from a list of language and their probability"""
        most_probable_langauge = all_language_probs[0][0]

        for langauge, prob in all_language_probs:
            if langauge in allowed_lanague:
                return langauge

        return most_probable_langauge

    def transcribe_to_segment(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: LanguageType = None,
        batch: bool = False,
        batch_size: int = 8,
    ) -> List[Segment]:
        """
        basic function to transcribe audio to segment data type

        Example of segment data:
        Segment(id=1, seek=0, start=0.0, end=4.0, text=' Then you g',
        tokens=[50365, 1396, 291, ],
        avg_logprob=-0.4401855356991291,
        compression_ratio=1.0185185185185186,
        no_speech_prob=0.06640625,
        words=None,
        temperature=0.0)
        """
        if language:
            logger.info(f"language is set to {language}")

        # using default parameters mostly
        language = language if language else self.original_language
        if isinstance(language, list) and len(language) == 1:
            language = language[0]

        if language == "auto":
            language = None

        # transcription for multilinguage with restrictred set of lanaguage
        if isinstance(language, list) and len(language) > 1:
            _language, _language_probability, all_language_probs = self.model.detect_language(audio=audio)
            logger.info("allowed language list: %s", language)
            if _language in language:
                most_probable_language = _language
            else:
                most_probable_language = self._filter_lanague(language, all_language_probs)

            segments, info = self.model.transcribe(
                _type="single", # only work with single
                audio=audio,
                language=most_probable_language,
                multilingual=False if language else True,
                task="transcribe",
                initial_prompt=init_prompt,
                beam_size=5,                        # tested: beam_size=5 is faster and better
                word_timestamps=True,               # timestamp is used for sequence matching when streaming
                condition_on_previous_text=True,
                temperature=temperature,
                vad_filter=True,
                repetition_penalty=1.0,
                **self.transcribe_kargs
            )
        else:  # all other scenario
            kwargs_args = {
                "_type": "single",
                "audio": audio,
                "language": language,
                "multilingual": False if language else True,
                "task": "transcribe",
                "initial_prompt": init_prompt,
                "beam_size": 5,  # tested: beam_size=5 is faster and better
                "word_timestamps": True,  # timestamp is used for sequence matching when streaming
                "condition_on_previous_text": True,
                "temperature": temperature,
                "vad_filter": True,
                "repetition_penalty": 1.0,
                ** self.transcribe_kargs
            }

            if batch:
                kwargs_args.update(
                    {
                        "_type": "batch",
                        "batch_size": batch_size if batch else 8,
                    }
                )

            segments, info = self.model.transcribe(**kwargs_args)

        logger.debug(info)  # info contains language detection result

        return list(segments)

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        init_prompt: str = "",
        temperature: float = 0.0,
        language: LanguageType = None,
        include_segments: bool = False,
        batch: bool = False,
    ) -> TranscriptionResponse:
        """
        similar to transcribe_to_segment, however, will concat all the words into the full text
        """

        # using default parameters mostly
        segments = self.transcribe_to_segment(
            audio,
            init_prompt=init_prompt,
            temperature=temperature,
            language=language,
            batch=batch
        )
        segments = list(segments)   # convert iterable to list

        transcribed_text = self.segment_to_long_text(segments)

        if not include_segments:
            return TranscriptionResponse(text=transcribed_text)
        else:
            # convert to dictionary for segment
            segments_dict = [dataclasses.asdict(segment) for segment in segments]

            return TranscriptionResponse(
                text=transcribed_text,
                segments=segments_dict    # convert to dictionary
            )


    def segment_to_timestamped_words(self, segments: List[Segment]) -> List[Tuple]:
        """
        Converts a list of segments into a list of word-level timestamps.

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
                output.append((word.start, word.end, word.word))
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


    def segments_end_ts(self, res):
        """
        return a list of ending time stamp from a list of segments
        """
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"