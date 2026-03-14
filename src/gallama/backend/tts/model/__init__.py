try:
    from .kokoro import TTSKokoro
except ModuleNotFoundError:
    TTSKokoro = None

# currently GPT Sovit is not hard dependency
try:
    from .gpt_sovits import TTS_GPT_SoVITS
except ModuleNotFoundError:
    TTS_GPT_SoVITS = None
