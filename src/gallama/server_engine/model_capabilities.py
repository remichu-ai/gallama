from __future__ import annotations

from typing import Any, Callable, Iterable


def _normalize_modalities(values: Any) -> set[str]:
    if not values:
        return set()

    if isinstance(values, str):
        return {values}

    if isinstance(values, (list, tuple, set)):
        return {str(value) for value in values if value}

    return set()


def _default_model_type_loader(model_id: str) -> str | None:
    try:
        from transformers import AutoConfig
    except Exception:
        return None

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None

    return getattr(config, "model_type", None)


def _default_vision_token_resolver(model_type: str | None) -> str | None:
    if not model_type:
        return None

    try:
        from gallama.backend.llm.prompt_engine.model_special_tag import resolve_vision_token
    except Exception:
        return None

    return resolve_vision_token(model_type=model_type, tokenizer=None)


def _looks_like_vision_prompt_template(prompt_template: str | None) -> bool:
    if not prompt_template:
        return False

    normalized = prompt_template.strip().lower()
    return normalized in {
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "glm4v",
        "vision",
    } or "vision" in normalized


def _looks_like_vision_model_name(model_name: str | None) -> bool:
    if not model_name:
        return False

    normalized = model_name.strip().lower()
    return any(marker in normalized for marker in ("-vl", "_vl", "vision", "glm4v"))


def infer_model_modalities_fallback(
    *,
    model_name: str | None,
    model_id: str | None,
    backend: str | None,
    prompt_template: str | None,
    config_records: Iterable[dict[str, Any] | None] = (),
    model_type_loader: Callable[[str], str | None] | None = None,
    vision_token_resolver: Callable[[str | None], str | None] | None = None,
) -> list[str]:
    """
    Best-effort modality inference when the worker did not report runtime modalities.

    Priority:
    1. Explicit config/default-model-list modalities.
    2. Known backend behavior.
    3. Prompt-engine style vision inference from model metadata.
    """
    modalities: set[str] = set()

    for record in config_records:
        if not isinstance(record, dict):
            continue
        modalities.update(_normalize_modalities(record.get("modalities")))

    if modalities:
        return sorted(modalities)

    if backend == "llama_cpp_server":
        modalities.add("image")

    if model_id:
        loader = model_type_loader or _default_model_type_loader
        resolver = vision_token_resolver or _default_vision_token_resolver
        model_type = loader(model_id)
        if resolver(model_type):
            modalities.add("image")

    if not modalities and _looks_like_vision_prompt_template(prompt_template):
        modalities.add("image")

    if not modalities and _looks_like_vision_model_name(model_name):
        modalities.add("image")

    return sorted(modalities)
