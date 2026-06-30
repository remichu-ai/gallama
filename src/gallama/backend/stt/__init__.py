try:
    from .faster_whisper.model import ASRFasterWhisper
except ModuleNotFoundError:
    ASRFasterWhisper = None

try:
    from .mlx_whisper.model import ASRMLXWhisper
except ModuleNotFoundError:
    ASRMLXWhisper = None

try:
    from .nemo_asr.model import ASRNeMo
except ModuleNotFoundError:
    ASRNeMo = None


from .asr_processor import ASRProcessor
