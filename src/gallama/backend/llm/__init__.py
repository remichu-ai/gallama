
from .thinking_template import THINKING_TEMPLATE

# selective import as not all backend available on all platform
try:
    from .engine import ModelExllama
except ImportError:
    ModelExllama = None

try:
    from .engine import ModelLlamaCpp
except ImportError:
    ModelLlamaCpp = None

try:
    from .engine import ModelTransformers
except ImportError:
    ModelTransformers = None

# try:
from .engine import ModelMLXVLM
# except ImportError:
#     ModelMLXVLM = None