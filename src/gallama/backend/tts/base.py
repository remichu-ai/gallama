from typing import Literal
from abc import ABC, abstractmethod
from typing import Dict, List

from gallama.data_classes import (
    ModelParser
)


class TTSBase(ABC):
    """this is the base interface for all TTS models"""

    def __init__(self,
        model_spec: ModelParser,
        model_config: Dict,
    ):

        self.model_name = model_spec.model_name or model_config["model_name"]

        # backend specific arguments
        self.backend_extra_args = model_config.get("backend_extra_args") or {}



    async def text_to_speech(
        self,
        text: str,
        language:str = "auto",
        stream: bool = False    # non stream return numpy array of whole speech, True will return iterator instead
    ):
        pass

