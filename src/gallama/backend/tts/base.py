from typing import Literal
from abc import ABC, abstractmethod
from typing import Dict, List

from gallama.data_classes import (
    ModelSpec
)


class TTSBase(ABC):
    """this is the base interface for all TTS models"""

    def __init__(self, model_spec: ModelSpec):
        self.model_name = model_spec.model_name
        # backend specific arguments
        self.backend_extra_args = model_spec.backend_extra_args


    async def text_to_speech(
        self,
        text: str,
        language:str = "auto",
        stream: bool = False    # non stream return numpy array of whole speech, True will return iterator instead
    ):
        pass

