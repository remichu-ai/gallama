from .by_model import (
    gpt_oss,
    qwen3_moe,
    qwen35,
    minimax,
    glm4,
    ministral3
)

# the key here should match the model_type field from transformers.AutoConfig
MODEL_SPECIAL_TAG = {
    "gpt_oss": gpt_oss,
    "qwen2": qwen3_moe,
    "qwen2_5_vl": qwen3_moe,
    "qwen3": qwen3_moe,
    "qwen3_moe":  qwen3_moe,
    "qwen3_next": qwen3_moe,
    "qwen3_vl": qwen3_moe,
    "qwen3_vl_moe": qwen3_moe,
    "qwen3_5": qwen35,
    "qwen3_5_moe": qwen35,
    "minimax": minimax,
    "minimax_m2": minimax,
    "glm4": glm4,
    "glm4v_moe": glm4,
    "glm4v": glm4,
    "glm4_moe": glm4,
    "ministral3": ministral3,
    "mistral3": ministral3,
    "mistral4": ministral3,
    "step3p5": qwen35,
    "nemotron_h": qwen35,
}

# to add on to the tokenizer.eos_token for model that having issue of wrong template
MODEL_EOS_TOKEN = {
    "gpt_oss": ["<|call|>", "<|return|>"],
    "glm4":  ["<|user|>", "<|observation|>"],
    "glm4v":  ["<|user|>", "<|observation|>"],
    "glm4v_moe":  ["<|user|>", "<|observation|>"],
    "glm4_moe":  ["<|user|>", "<|observation|>"],
    "mistral3": ["</s>"],
    "mistral4": ["</s>"]
}

MODEL_VISION_TOKEN = {
    "qwen2_5_vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "qwen3_vl": "<|vision_start|><|image_pad|><|vision_end|>",
    "qwen3_vl_moe": "<|vision_start|><|image_pad|><|vision_end|>",
    "glm4v": "<|begin_of_image|><|image|><|end_of_image|>",
    "glm4v_moe": "<|begin_of_image|><|image|><|end_of_image|>"
}

_VISION_TOKEN_PATTERNS = (
    ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
    ("<|vision_bos|>", "<|IMAGE|>", "<|vision_eos|>"),
    ("<|begin_of_image|>", "<|image|>", "<|end_of_image|>"),
    ("<start_of_image>", "<image_soft_token>", "<end_of_image>"),
    ("", "<|image|>", ""),
)


def _collect_special_tokens(tokenizer) -> set[str]:
    if tokenizer is None:
        return set()

    special_tokens: set[str] = set()

    def _add(token):
        if not token:
            return
        if isinstance(token, str):
            special_tokens.add(token)
            return
        content = getattr(token, "content", None)
        if isinstance(content, str):
            special_tokens.add(content)
            return
        try:
            special_tokens.add(str(token))
        except Exception:
            return

    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        _add(token)

    for token in (getattr(tokenizer, "special_tokens_map", {}) or {}).values():
        if isinstance(token, list):
            for item in token:
                _add(item)
        else:
            _add(token)

    for token in (getattr(tokenizer, "added_tokens_decoder", {}) or {}).values():
        _add(token)

    return special_tokens


def infer_vision_token(tokenizer) -> str | None:
    special_tokens = _collect_special_tokens(tokenizer)
    if not special_tokens:
        return None

    for start_token, image_token, end_token in _VISION_TOKEN_PATTERNS:
        has_start = not start_token or start_token in special_tokens
        has_image = image_token in special_tokens
        has_end = not end_token or end_token in special_tokens
        if has_start and has_image and has_end:
            return f"{start_token}{image_token}{end_token}"

    return None


def resolve_vision_token(model_type: str | None, tokenizer=None) -> str | None:
    return MODEL_VISION_TOKEN.get(model_type) or infer_vision_token(tokenizer)
