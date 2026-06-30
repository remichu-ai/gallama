from importlib import import_module

from .format_enforcer import SGLangFormatter

_MODEL_IMPORTS = {
    "ModelExllama": ".engine.exllama",
    "ModelExllamaV3": ".engine.exllamav3",
    "ModelLlamaCpp": ".engine.llamacpp",
    "ModelLlamaCppServer": ".engine.llamacpp_server",
    "ModelIKLlama": ".engine.ik_llama",
    "ModelTransformers": ".engine.transformers",
    "ModelMLXVLM": ".engine.mlx_vllm",
    "ModelSGLang": ".engine.sglang",
    "ModelVLLM": ".engine.vllm",
}

__all__ = ["SGLangFormatter", *_MODEL_IMPORTS]


def __getattr__(name):
    module_path = _MODEL_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module(module_path, __name__)
        value = getattr(module, name)
    except ImportError:
        value = None

    globals()[name] = value
    return value
