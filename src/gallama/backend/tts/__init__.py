from .model.kokoro import TTSKokoro

try:
    from .model.gpt_sovits import TTS_GPT_SoVITS
except ModuleNotFoundError:
    pass