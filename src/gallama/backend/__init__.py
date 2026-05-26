from importlib import import_module


__all__ = [
    "EmbeddingModel",
    "RerankerModel",
    "ModelExllama",
    "ModelLlamaCpp",
    "ModelTransformers",
]


def __getattr__(name):
    if name == "EmbeddingModel":
        from .embedding.embedding import EmbeddingModel

        return EmbeddingModel

    if name == "RerankerModel":
        from .reranker.reranker import RerankerModel

        return RerankerModel

    if name in {"ModelExllama", "ModelLlamaCpp", "ModelTransformers"}:
        llm = import_module(f"{__name__}.llm")

        return getattr(llm, name, None)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
