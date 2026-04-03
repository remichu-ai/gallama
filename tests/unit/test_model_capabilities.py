import importlib.util
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODULE_PATH = os.path.join(SRC_DIR, "gallama", "server_engine", "model_capabilities.py")
MODULE_SPEC = importlib.util.spec_from_file_location("model_capabilities_test_module", MODULE_PATH)
MODEL_CAPABILITIES_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(MODEL_CAPABILITIES_MODULE)

infer_model_modalities_fallback = MODEL_CAPABILITIES_MODULE.infer_model_modalities_fallback


def test_infer_model_modalities_prefers_explicit_config_modalities():
    modalities = infer_model_modalities_fallback(
        model_name="qwen-2.5-VL-7B",
        model_id="/models/qwen",
        backend="exllama",
        prompt_template="Qwen2-VL",
        config_records=[{"modalities": ["video"]}],
    )

    assert modalities == ["video"]


def test_infer_model_modalities_uses_prompt_engine_style_model_type_fallback():
    modalities = infer_model_modalities_fallback(
        model_name="unknown",
        model_id="/models/qwen3-vl",
        backend="transformers",
        prompt_template=None,
        model_type_loader=lambda model_id: "qwen3_vl",
        vision_token_resolver=lambda model_type: "<|vision_start|><|image_pad|><|vision_end|>",
    )

    assert modalities == ["image"]


def test_infer_model_modalities_uses_prompt_template_heuristic_last():
    modalities = infer_model_modalities_fallback(
        model_name="custom-vlm",
        model_id=None,
        backend="transformers",
        prompt_template="Qwen2-VL",
    )

    assert modalities == ["image"]


def test_infer_model_modalities_knows_llama_cpp_server_is_multimodal():
    modalities = infer_model_modalities_fallback(
        model_name="llava-server",
        model_id=None,
        backend="llama_cpp_server",
        prompt_template=None,
    )

    assert modalities == ["image"]
