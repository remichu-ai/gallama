import ast
import os
from types import SimpleNamespace
from typing import Dict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODULE_PATH = os.path.join(
    ROOT_DIR,
    "src",
    "gallama",
    "backend",
    "llm",
    "engine",
    "exllamav3",
    "exllamav3.py",
)


def _load_generator_helpers():
    with open(MODULE_PATH, encoding="utf-8") as f:
        source = f.read()

    module_ast = ast.parse(source, filename=MODULE_PATH)
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_is_truthy",
            "_normalize_generator_kwargs",
            "_align_cache_size",
            "_is_insufficient_vram_error",
            "_normalize_reserve_vram",
            "_auto_use_vram_for_existing_allocations",
            "_resolve_load_kwargs",
        }
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {"Dict": Dict, "List": list}
    exec(compile(helper_module, MODULE_PATH, "exec"), namespace)
    return (
        namespace["_normalize_generator_kwargs"],
        namespace["_align_cache_size"],
        namespace["_is_insufficient_vram_error"],
        namespace["_normalize_reserve_vram"],
        namespace["_auto_use_vram_for_existing_allocations"],
        namespace["_resolve_load_kwargs"],
    )


def _load_model_method_ast(method_name):
    with open(MODULE_PATH, encoding="utf-8") as f:
        source = f.read()

    module_ast = ast.parse(source, filename=MODULE_PATH)
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelExllamaV3":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"ModelExllamaV3.{method_name} not found")


(
    normalize_generator_kwargs,
    align_cache_size,
    is_insufficient_vram_error,
    normalize_reserve_vram,
    auto_use_vram_for_existing_allocations,
    resolve_load_kwargs,
) = _load_generator_helpers()


def test_normalize_generator_kwargs_defaults():
    result = normalize_generator_kwargs({})
    assert result["max_chunk_size"] == 2048
    assert result["max_batch_size"] == 4

    result_none = normalize_generator_kwargs(None)
    assert result_none["max_chunk_size"] == 2048
    assert result_none["max_batch_size"] == 4


def test_normalize_generator_kwargs_preserves_explicit_values():
    normalized = normalize_generator_kwargs({"max_chunk_size": "2048", "max_batch_size": "32"})

    assert normalized["max_chunk_size"] == 2048
    assert normalized["max_batch_size"] == 32


def test_align_cache_size_rounds_up_to_page_size():
    assert align_cache_size(4097, 4097) == 4352
    assert align_cache_size(4096, 4097) == 4352
    assert align_cache_size(None, 4097) == 4352


def test_is_insufficient_vram_error_matches_exllamav3_and_cuda_oom_messages():
    assert is_insufficient_vram_error(RuntimeError("Insufficient VRAM in split for model and cache"))
    assert is_insufficient_vram_error(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not is_insufficient_vram_error(RuntimeError("tokenizer failed"))


def test_normalize_reserve_vram_defaults_to_zero_point_eight_gb_for_gpu_zero():
    assert normalize_reserve_vram(None, 3) == [0.8, 0.4, 0.4]


def test_normalize_reserve_vram_pads_short_lists_with_zero():
    assert normalize_reserve_vram([1.0], 3) == [1.0, 0.0, 0.0]


def test_auto_use_vram_for_existing_allocations_uses_total_minus_reserve():
    gib = 1024 ** 3

    class FakeCuda:
        @staticmethod
        def get_device_properties(device_idx):
            return SimpleNamespace(total_memory=(100 - device_idx * 10) * gib)

    auto_use_vram_for_existing_allocations.__globals__["torch"] = SimpleNamespace(cuda=FakeCuda())

    assert auto_use_vram_for_existing_allocations(None, 2) == [99.2, 89.6]
    assert auto_use_vram_for_existing_allocations([1.0, -1.0], 2) == [99.0, 0.0]


def test_resolve_load_kwargs_omits_reserve_per_device_for_auto_mode():
    resolved = resolve_load_kwargs("auto", 0.75, False, 2)

    assert resolved["tensor_p"] is False
    assert "reserve_per_device" not in resolved
    assert "use_per_device" not in resolved


def test_resolve_load_kwargs_rejects_reserve_with_explicit_gpu_split():
    try:
        resolve_load_kwargs([20.0, 20.0], 0.5, False, 2)
    except ValueError as exc:
        assert "does not support `reserve_vram` together with an explicit `gpus` split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible gpus/reserve_vram combination")


def test_exllamav3_loads_vision_before_text_cache_autosplit():
    load_model_exllama = _load_model_method_ast("load_model_exllama")

    vision_call_line = None
    model_load_line = None
    for node in ast.walk(load_model_exllama):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_load_vision_processor":
                vision_call_line = node.lineno
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "model"
            ):
                model_load_line = node.lineno

    assert vision_call_line is not None
    assert model_load_line is not None
    assert vision_call_line < model_load_line


def test_exllamav3_resets_cuda_memory_fraction_after_load_failures():
    load_model_exllama = _load_model_method_ast("load_model_exllama")
    load_model = _load_model_method_ast("load_model")

    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reset_cuda_memory_fraction"
        for node in ast.walk(load_model_exllama)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_reset_cuda_memory_fraction"
        for node in ast.walk(load_model)
    )
