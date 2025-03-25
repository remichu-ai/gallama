from .kokoro import TTSKokoro

# currently GPT Sovit is not hard dependency
try:
    from .gpt_sovits import TTS_GPT_SoVITS
except ModuleNotFoundError:
    pass
