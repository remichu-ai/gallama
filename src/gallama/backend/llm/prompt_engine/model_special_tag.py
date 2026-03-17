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
