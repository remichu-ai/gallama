from typing import Literal
from abc import ABC, abstractmethod
from ...data_classes import ModelSpec

class ASRBase(ABC):

    def __init__(
        self,
        model_spec: ModelSpec,
        # language: str="auto",
        # modelsize=None,
        # cache_dir=None,
        # model_dir=None,
        # quant: Literal["8.0", "16.0"] = "16.0",
        # device: TODO to implement
    ):

        if model_spec.language == "auto":
            self.original_language = None
        else:
            self.original_language = model_spec.language

        self.quant = model_spec.quant

        self.sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                         # "" for faster-whisper because it emits the spaces when neeeded)

        self.transcribe_kargs = {}
        self.model = self.load_model(model_spec)

    @abstractmethod
    def load_model(self, model_spec: ModelSpec):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def segment_to_timestamped_words(self, segments):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def transcribe(self, audio, init_prompt: str = "", temperature: float = 0.0):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def transcribe_to_segment(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    @abstractmethod
    def segments_end_ts(self, res):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")