from .by_model import (
    qwen3_moe,
    minimax,
    glm4,
    ministral3
)

# the key here should match the model_type field from transformers.AutoConfig
MODEL_SPECIAL_TAG = {
    "qwen3_moe":  qwen3_moe,
    "minimax": minimax,
    "minimax_m2": minimax,
    "glm4v_moe": glm4,
    "glm4_moe": glm4,
    "ministral3": ministral3,
    "mistral3": ministral3,
}

# to add on to the tokenizer.eos_token for model that having issue of wrong template
MODEL_EOS_TOKEN = {
    "glm4v_moe":  ["<|user|>", "<|observation|>"],
    "glm4_moe":  ["<|user|>", "<|observation|>"],
    "mistral3": ["</s>"]
}