import ast
import os
from types import SimpleNamespace
from typing import Dict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODULE_PATH = os.path.join(
    ROOT_DIR,
    "src",
    "gallama",
    "server_engine",
    "model_management.py",
)


def _load_status_helpers():
    with open(MODULE_PATH, encoding="utf-8") as f:
        source = f.read()

    module_ast = ast.parse(source, filename=MODULE_PATH)
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"format_cuda_visible_devices", "log_model_status"}
    ]
    helper_module = ast.Module(body=helper_nodes, type_ignores=[])
    namespace = {
        "Dict": Dict,
        "ModelInfo": object,
        "basic_log_extra": lambda: {},
        "logger": None,
        "get_gpu_memory_info": lambda: "",
    }
    exec(compile(helper_module, MODULE_PATH, "exec"), namespace)
    return namespace


HELPERS = _load_status_helpers()
format_cuda_visible_devices = HELPERS["format_cuda_visible_devices"]
log_model_status = HELPERS["log_model_status"]


class _CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)


def test_format_cuda_visible_devices_reports_logical_to_physical_mapping():
    assert (
        format_cuda_visible_devices("3,1,0,2")
        == "cuda:0->GPU 3, cuda:1->GPU 1, cuda:2->GPU 0, cuda:3->GPU 2"
    )


def test_log_model_status_includes_effective_cuda_visible_devices():
    HELPERS["get_gpu_memory_info"] = lambda: "GPU 0: Used:  1.0GB, Free:  2.0GB, Total:  3.0GB"

    models = {
        "qwen3.5-397B": SimpleNamespace(
            instances=[
                SimpleNamespace(
                    model_name="qwen3.5-397B",
                    port=8001,
                    cuda_visible_devices="3,1,0,2",
                )
            ]
        )
    }
    logger = _CaptureLogger()

    log_model_status(models, custom_logger=logger)

    assert len(logger.messages) == 1
    message = logger.messages[0]
    assert "Effective CUDA_VISIBLE_DEVICES per instance" in message
    assert "qwen3.5-397B:8001" in message
    assert "cuda:0->GPU 3, cuda:1->GPU 1, cuda:2->GPU 0, cuda:3->GPU 2" in message
    assert "Physical GPU Memory Information (nvidia-smi order)" in message
