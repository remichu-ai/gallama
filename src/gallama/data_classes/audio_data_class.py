from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal, Dict, Any, Tuple
from fastapi import Query


_LANGUAGE_CODES = [
    "auto",
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
]

# Create a reusable type for language
LanguageType = Optional[Union[List[Literal[*_LANGUAGE_CODES]], Literal[*_LANGUAGE_CODES]]]


# class for audio api call
class TranscriptionResponse(BaseModel):
    task: Optional[str] = Field(default="transcribe", description="The task performed.")
    language: LanguageType = Field(
        default=None,
        description="The detected or specified language of the audio."
    )
    duration: Optional[float] = Field(default=None, description="The duration of the audio in seconds.")
    text: str = Field(description="The transcription text.")
    # TODO to move to use only Dict
    words: Optional[List[Union[Dict[Any, Any], Tuple]]] = Field(default=None, description="Word-level timestamps, if requested.")
    segments: Optional[List[Dict[Any, Any]]] = Field(default=None, description="Segment-level timestamps, if requested.")


# class for coding
class TimeStampedWord(BaseModel):
    pass


# Define supported models, voices, and formats
SUPPORTED_MODELS = {"tts-1", "tts-1-hd"}
SUPPORTED_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
DEFAULT_FORMAT = "wav"

# Pydantic model for request validation for text to speech
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"
    response_format: str = Query(DEFAULT_FORMAT, enum=list(SUPPORTED_FORMATS))
    speed: float = Query(1.0, ge=0.25, le=4.0)



