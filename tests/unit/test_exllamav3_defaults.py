import ast
import os
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
            "_normalize_reserve_vram",
            "_resolve_load_kwargs",
        }
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {"Dict": Dict, "List": list}
    exec(compile(helper_module, MODULE_PATH, "exec"), namespace)
    return (
        namespace["_normalize_generator_kwargs"],
        namespace["_normalize_reserve_vram"],
        namespace["_resolve_load_kwargs"],
    )


normalize_generator_kwargs, normalize_reserve_vram, resolve_load_kwargs = _load_generator_helpers()


def test_normalize_generator_kwargs_defaults_prompt_chunk_size_to_4096():
    assert normalize_generator_kwargs({})["max_chunk_size"] == 4096
    assert normalize_generator_kwargs(None)["max_chunk_size"] == 4096


def test_normalize_generator_kwargs_preserves_explicit_prompt_chunk_size():
    normalized = normalize_generator_kwargs({"max_chunk_size": "2048", "max_batch_size": "32"})

    assert normalized["max_chunk_size"] == 2048
    assert normalized["max_batch_size"] == 32


def test_normalize_reserve_vram_defaults_to_zero_point_eight_gb_for_gpu_zero():
    assert normalize_reserve_vram(None, 3) == [0.8, 0.4, 0.4]


def test_normalize_reserve_vram_pads_short_lists_with_zero():
    assert normalize_reserve_vram([1.0], 3) == [1.0, 0.0, 0.0]


def test_resolve_load_kwargs_uses_reserve_per_device_for_auto_mode():
    resolved = resolve_load_kwargs("auto", 0.75, False, 2)

    assert resolved["reserve_per_device"] == [0.75, 0.75]
    assert resolved["tensor_p"] is False
    assert "use_per_device" not in resolved


def test_resolve_load_kwargs_rejects_reserve_with_explicit_gpu_split():
    try:
        resolve_load_kwargs([20.0, 20.0], 0.5, False, 2)
    except ValueError as exc:
        assert "does not support `reserve_vram` together with an explicit `gpus` split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible gpus/reserve_vram combination")
