try:
    from .kokoro import TTSKokoro
except ModuleNotFoundError:
    TTSKokoro = None
