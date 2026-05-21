import asyncio

import httpx
import pytest

from gallama.backend.llm.engine.ik_llama.ik_llama import ModelIKLlama
from gallama.backend.llm.engine.llamacpp_server.llamacpp_server import ModelLlamaCppServer


class _AsyncBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"error":"bad payload"}'

    async def aclose(self):
        return None


def test_read_error_response_text_reads_unconsumed_async_stream():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://127.0.0.1:8080/completion"),
        stream=_AsyncBody(),
    )

    with pytest.raises(httpx.ResponseNotRead):
        _ = response.text

    detail = asyncio.run(ModelLlamaCppServer._read_error_response_text(response))

    assert detail == '{"error":"bad payload"}'


def test_rewrite_multimodal_prompt_string_replaces_qwen_marker():
    server = ModelLlamaCppServer.__new__(ModelLlamaCppServer)
    server.multimodal_marker = "<__media__>"
    server.prompt_eng = None

    prompt = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>What animal is in this image?<|im_end|>\n"
    )

    rewritten = server._rewrite_multimodal_prompt_string(
        prompt,
        multimodal_count=1,
        vision_token="<|vision_start|><|image_pad|><|vision_end|>",
    )

    assert rewritten.count("<__media__>") == 1
    assert "<|vision_start|><|image_pad|><|vision_end|>" not in rewritten


def test_rewrite_multimodal_prompt_string_is_noop_without_configured_marker():
    server = ModelLlamaCppServer.__new__(ModelLlamaCppServer)
    server.multimodal_marker = None
    server.prompt_eng = None

    prompt = "<|vision_start|><|image_pad|><|vision_end|>Describe this image."

    rewritten = server._rewrite_multimodal_prompt_string(prompt, multimodal_count=1)

    assert rewritten == prompt


def test_ik_llama_applies_default_multimodal_marker():
    resolved = ModelIKLlama.apply_backend_defaults({})

    assert resolved["multimodal_marker"] == "<__media__>"


def test_llama_cpp_server_applies_default_multimodal_marker():
    resolved = ModelLlamaCppServer.apply_backend_defaults({})

    assert resolved["multimodal_marker"] == "<__media__>"


def test_llama_cpp_server_uses_model_concurrency_for_parallel_flag():
    server = ModelLlamaCppServer.__new__(ModelLlamaCppServer)
    server.start_up_cmd = "llama-server"
    server.model_id = "/models/model.gguf"
    server.server_host = "127.0.0.1"
    server.server_port = 8080
    server.max_seq_len = 4096
    server.max_concurrent_requests = 1
    server.backend_extra_args = {}
    server.start_up_extra = None

    cmd = server._build_startup_command()

    assert "--parallel" in cmd
    assert cmd[cmd.index("--parallel") + 1] == "1"


def test_llama_cpp_server_parallel_backend_arg_overrides_model_concurrency():
    server = ModelLlamaCppServer.__new__(ModelLlamaCppServer)
    server.start_up_cmd = "llama-server"
    server.model_id = "/models/model.gguf"
    server.server_host = "127.0.0.1"
    server.server_port = 8080
    server.max_seq_len = 4096
    server.max_concurrent_requests = 1
    server.backend_extra_args = {"parallel": 3}
    server.start_up_extra = None

    cmd = server._build_startup_command()

    assert cmd[cmd.index("--parallel") + 1] == "3"


def test_ik_llama_preserves_explicit_multimodal_marker_override():
    resolved = ModelIKLlama.apply_backend_defaults({"multimodal_marker": "<custom>"})

    assert resolved["multimodal_marker"] == "<custom>"
