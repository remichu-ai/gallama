try:
    from .model.kokoro import TTSKokoro
except ModuleNotFoundError:
    TTSKokoro = None
