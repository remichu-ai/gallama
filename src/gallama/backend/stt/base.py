from typing import Literal
from abc import ABC, abstractmethod

class ASRBase(ABC):

    def __init__(
        self,
        language: str="auto",
        modelsize=None,
        cache_dir=None,
        model_dir=None,
        quant: Literal["8.0", "16.0"] = "16.0",
        # device: TODO to implement
    ):

        if language == "auto":
            self.original_language = None
        else:
            self.original_language = language

        self.quant = quant

        self.sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                         # "" for faster-whisper because it emits the spaces when neeeded)

        self.transcribe_kargs = {}
        self.model = self.load_model(modelsize, cache_dir, model_dir, quant)

    @abstractmethod
    def load_model(self, modelsize, cache_dir):
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