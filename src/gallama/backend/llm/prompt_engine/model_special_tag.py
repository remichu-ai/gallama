from .by_model import (
    qwen3_moe,
    minimax
)

# the key here should match the model_type field from transformers.AutoConfig
MODEL_SPECIAL_TAG = {
    "qwen3_moe":  qwen3_moe,
    "minimax": minimax,
}