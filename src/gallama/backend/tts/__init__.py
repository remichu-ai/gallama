try:
    from .model.kokoro import TTSKokoro
except ModuleNotFoundError:
    TTSKokoro = None

try:
    from .model.gpt_sovits import TTS_GPT_SoVITS
except ModuleNotFoundError:
    TTS_GPT_SoVITS = None
