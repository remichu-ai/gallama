"""Unit tests for the ExLlamaV3 segmented-topology / placement config plumbing.

These cover the helpers in src/gallama/backend/llm/engine/exllamav3/exllamav3.py that translate
``backend_extra_args`` entries (segment_topology / segment_topology_yaml / placement_yaml) into the
``model.load(...)`` kwargs consumed by the minimax-m3 ExLlamaV3 fork. Helpers are extracted by AST
so the tests do not import torch/exllamav3 or require a GPU.
"""

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

_TOPOLOGY_HELPERS = {
    "_gallama_config_dir",
    "_resolve_topology_path",
    "_materialize_segment_topology",
    "_cleanup_topology_temp_file",
    "_apply_exllamav3_tp_load_options",
}


def _load_topology_helpers():
    with open(MODULE_PATH, encoding="utf-8") as f:
        source = f.read()

    module_ast = ast.parse(source, filename=MODULE_PATH)
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in _TOPOLOGY_HELPERS
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {"Dict": Dict, "os": os}
    exec(compile(helper_module, MODULE_PATH, "exec"), namespace)
    return namespace


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


ns = _load_topology_helpers()
gallama_config_dir = ns["_gallama_config_dir"]
resolve_topology_path = ns["_resolve_topology_path"]
materialize_segment_topology = ns["_materialize_segment_topology"]
cleanup_topology_temp_file = ns["_cleanup_topology_temp_file"]
apply_exllamav3_tp_load_options = ns["_apply_exllamav3_tp_load_options"]


# --------------------------------------------------------------------------------------------------
# _resolve_topology_path
# --------------------------------------------------------------------------------------------------

def test_resolve_topology_path_keeps_absolute_paths(tmp_path):
    absolute = str(tmp_path / "topo.yaml")
    assert resolve_topology_path(absolute) == absolute


def test_resolve_topology_path_keeps_existing_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cwd_topo.yaml").write_text("segments: []")
    assert resolve_topology_path("cwd_topo.yaml") == "cwd_topo.yaml"


def test_resolve_topology_path_falls_back_to_gallama_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    (tmp_path / "stored_topo.yaml").write_text("segments: []")
    # The file does not exist in CWD, only under the gallama config dir.
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_topology_path("stored_topo.yaml") == os.path.join(str(tmp_path), "stored_topo.yaml")


def test_resolve_topology_path_returns_input_when_unresolvable():
    assert resolve_topology_path("does_not_exist_anywhere.yaml") == "does_not_exist_anywhere.yaml"


# --------------------------------------------------------------------------------------------------
# _materialize_segment_topology
# --------------------------------------------------------------------------------------------------

def test_materialize_segment_topology_writes_valid_yaml_round_trip():
    inline = {
        "segments": [
            {"kind": "tp", "name": "front", "devices": [0, 1], "layers": [0, 40],
             "output_device": 0, "options": {"moe_tensor_split": True, "tp_backend": "nccl"}},
            {"kind": "ls", "name": "tail", "devices": [2], "layers": [40, 60], "include_epilog": True},
        ]
    }
    path = materialize_segment_topology(inline)
    try:
        assert os.path.exists(path)
        import yaml
        with open(path) as f:
            loaded = yaml.safe_load(f)
        # Round-trips the structure the fork's parse_segment_topology expects.
        assert loaded == inline
        assert isinstance(loaded["segments"], list) and len(loaded["segments"]) == 2
        assert loaded["segments"][0]["layers"] == [0, 40]
        assert loaded["segments"][1]["include_epilog"] is True
    finally:
        cleanup_topology_temp_file(path)
        assert not os.path.exists(path)


def test_materialize_segment_topology_rejects_non_mapping():
    try:
        materialize_segment_topology(["not", "a", "mapping"])  # type: ignore[arg-type]
    except ValueError as exc:
        assert "segment_topology must be a mapping" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-mapping segment topology")


# --------------------------------------------------------------------------------------------------
# _apply_exllamav3_tp_load_options
# --------------------------------------------------------------------------------------------------

def _apply(extra, tensor_parallel=True):
    load_kwargs = {}
    apply_exllamav3_tp_load_options(load_kwargs, extra, tensor_parallel)
    return load_kwargs


def test_apply_passes_tp_backend_and_tp_options_through():
    load_kwargs = _apply({"tp_backend": "nccl", "tp_options": {"moe_tensor_split": True}})
    assert load_kwargs["tp_backend"] == "nccl"
    assert load_kwargs["tp_options"] == {"moe_tensor_split": True}


def test_apply_injects_top_level_placement_yaml_into_tp_options(tmp_path, monkeypatch):
    placement = tmp_path / "placement.yaml"
    placement.write_text("attn: auto")
    load_kwargs = _apply({"placement_yaml": str(placement)})
    assert load_kwargs["tp_options"] == {"placement_yaml": str(placement)}


def test_apply_keeps_existing_tp_options_when_injecting_placement_yaml(tmp_path):
    placement = tmp_path / "placement.yaml"
    placement.write_text("attn: auto")
    load_kwargs = _apply({"tp_options": {"moe_tensor_split": True}, "placement_yaml": str(placement)})
    assert load_kwargs["tp_options"] == {"moe_tensor_split": True, "placement_yaml": str(placement)}


def test_apply_rejects_conflicting_placement_yaml(tmp_path):
    placement = tmp_path / "placement.yaml"
    placement.write_text("attn: auto")
    try:
        _apply({
            "tp_options": {"placement_yaml": "/somewhere/else.yaml"},
            "placement_yaml": str(placement),
        })
    except ValueError as exc:
        assert "Conflicting placement_yaml" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting placement_yaml")


def test_apply_rejects_non_mapping_tp_options():
    try:
        _apply({"tp_options": ["nope"]})
    except ValueError as exc:
        assert "tp_options must be a mapping" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-mapping tp_options")


def test_apply_sets_segment_topology_yaml_from_inline(tmp_path, monkeypatch):
    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    inline = {"segments": [{"kind": "ls", "name": "all", "devices": [0], "layers": [0, 4],
                            "include_epilog": True}]}
    load_kwargs = _apply({"segment_topology": inline})
    path = load_kwargs["segment_topology_yaml"]
    assert path and os.path.exists(path)
    import yaml
    with open(path) as f:
        assert yaml.safe_load(f) == inline
    cleanup_topology_temp_file(path)


def test_apply_sets_segment_topology_yaml_from_file_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    topo = tmp_path / "topo.yaml"
    topo.write_text("segments: []")
    load_kwargs = _apply({"segment_topology_yaml": "topo.yaml"})
    assert load_kwargs["segment_topology_yaml"] == "topo.yaml"


def test_apply_rejects_both_inline_and_path_topology():
    try:
        _apply({"segment_topology": {"segments": []}, "segment_topology_yaml": "topo.yaml"})
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both topology forms are given")


def test_apply_requires_tensor_parallel_for_topology(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    topo = tmp_path / "topo.yaml"
    topo.write_text("segments: []")
    try:
        _apply({"segment_topology_yaml": "topo.yaml"}, tensor_parallel=False)
    except ValueError as exc:
        assert "tensor parallel" in str(exc)
    else:
        raise AssertionError("Expected ValueError when topology used without tensor parallel")


def test_apply_noop_for_plain_config():
    load_kwargs = _apply({})
    assert "tp_backend" not in load_kwargs
    assert "tp_options" not in load_kwargs
    assert "segment_topology_yaml" not in load_kwargs


# --------------------------------------------------------------------------------------------------
# load_model_exllama wiring (AST-level, mirrors test_exllamav3_defaults.py style)
# --------------------------------------------------------------------------------------------------

def test_load_model_exllama_invokes_topology_helper_before_model_load():
    method = _load_model_method_ast("load_model_exllama")

    helper_line = None
    model_load_line = None
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_apply_exllamav3_tp_load_options":
                helper_line = node.lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
        ):
            model_load_line = node.lineno

    assert helper_line is not None, "_apply_exllamav3_tp_load_options must be called in load_model_exllama"
    assert model_load_line is not None, "model.load(...) must be called in load_model_exllama"
    assert helper_line < model_load_line, "topology options must be resolved before model.load()"
