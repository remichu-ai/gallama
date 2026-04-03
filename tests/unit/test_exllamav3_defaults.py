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
        and node.name in {"_is_truthy", "_normalize_generator_kwargs"}
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {"Dict": Dict}
    exec(compile(helper_module, MODULE_PATH, "exec"), namespace)
    return namespace["_normalize_generator_kwargs"]


normalize_generator_kwargs = _load_generator_helpers()


def test_normalize_generator_kwargs_defaults_prompt_chunk_size_to_4096():
    assert normalize_generator_kwargs({})["max_chunk_size"] == 4096
    assert normalize_generator_kwargs(None)["max_chunk_size"] == 4096


def test_normalize_generator_kwargs_preserves_explicit_prompt_chunk_size():
    normalized = normalize_generator_kwargs({"max_chunk_size": "2048", "max_batch_size": "32"})

    assert normalized["max_chunk_size"] == 2048
    assert normalized["max_batch_size"] == 32
