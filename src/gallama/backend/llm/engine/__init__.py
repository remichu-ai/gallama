from importlib import import_module

_MODEL_IMPORTS = {
    "ModelExllama": ".exllama",
    "ModelLlamaCpp": ".llamacpp",
    "ModelLlamaCppServer": ".llamacpp_server",
    "ModelIKLlama": ".ik_llama",
    "ModelTransformers": ".transformers",
    "ModelMLXVLM": ".mlx_vllm",
    "ModelSGLang": ".sglang",
    "ModelExllamaV3": ".exllamav3",
    "ModelVLLM": ".vllm",
}

__all__ = list(_MODEL_IMPORTS)


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
