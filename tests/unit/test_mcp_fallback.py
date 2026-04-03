import asyncio
import logging
import os
import sys
import types


def _install_test_stubs():
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def device_count():
                return 0

        torch_stub.cuda = _Cuda()
        sys.modules["torch"] = torch_stub

    if "colorama" not in sys.modules:
        colorama_stub = types.ModuleType("colorama")
        colorama_stub.Fore = types.SimpleNamespace(
            CYAN="",
            GREEN="",
            YELLOW="",
            RED="",
            BLUE="",
        )
        colorama_stub.Back = types.SimpleNamespace(WHITE="")
        colorama_stub.Style = types.SimpleNamespace(RESET_ALL="")
        colorama_stub.init = lambda autoreset=True: None
        sys.modules["colorama"] = colorama_stub

    if "zmq" not in sys.modules:
        zmq_stub = types.ModuleType("zmq")

        class _Again(Exception):
            pass

        class _ZMQError(Exception):
            pass

        class _Socket:
            def setsockopt(self, *args, **kwargs):
                return None

            def connect(self, *args, **kwargs):
                return None

            def close(self):
                return None

            def send_json(self, *args, **kwargs):
                return None

        class _Context:
            def socket(self, *args, **kwargs):
                return _Socket()

            def term(self):
                return None

        zmq_stub.Context = _Context
        zmq_stub.PUSH = 0
        zmq_stub.LINGER = 0
        zmq_stub.NOBLOCK = 0
        zmq_stub.Again = _Again
        zmq_stub.error = types.SimpleNamespace(ZMQError=_ZMQError)
        sys.modules["zmq"] = zmq_stub

    if "fastapi" not in sys.modules:
        fastapi_stub = types.ModuleType("fastapi")
        fastapi_stub.Query = lambda default=None, **kwargs: default

        class _HTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class _Request:
            pass

        fastapi_stub.HTTPException = _HTTPException
        fastapi_stub.Request = _Request
        sys.modules["fastapi"] = fastapi_stub


_install_test_stubs()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
SRC_TESTS_DIR = os.path.join(ROOT_DIR, "src", "tests")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SRC_TESTS_DIR not in sys.path:
    sys.path.insert(0, SRC_TESTS_DIR)

from helpers.dummy_mcp_server import TEST_TOOL_NAME, run_dummy_mcp_server
from gallama.data_classes.data_class import ChatMLQuery
from gallama.remote_mcp.models import MCPResolvedTool, MCPServerConfig
from gallama.remote_mcp.orchestrator import _prepare_mcp_tools
from gallama.remote_mcp.runtime import MCPRuntimeError


def test_prepare_mcp_tools_skips_unavailable_servers_and_keeps_healthy_ones(caplog, monkeypatch):
    with run_dummy_mcp_server() as live_server:
        healthy_config = MCPServerConfig(name="healthy", url=live_server.url)
        failing_config = MCPServerConfig(name="local", url="http://127.0.0.1:1/mcp")

        async def fake_list_tools(self, server):
            if server.name == "local":
                raise MCPRuntimeError(
                    f"Unable to connect to MCP server '{server.name}' at {server.url}: connect failed"
                )
            return [
                MCPResolvedTool(
                    server_name=server.name,
                    tool_name=TEST_TOOL_NAME,
                    synthetic_name=f"mcp__{server.name}__{TEST_TOOL_NAME}",
                    description="Dummy tool",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]

        monkeypatch.setattr("gallama.remote_mcp.orchestrator.MCPRuntime.list_tools", fake_list_tools)
        caplog.set_level(logging.WARNING)

        async def _run():
            return await _prepare_mcp_tools(
                base_query=ChatMLQuery.model_validate(
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                    }
                ),
                mcp_servers=[failing_config, healthy_config],
            )

        runtime, discovered_tools, resolved_by_name, merged_tools = asyncio.run(_run())

        assert runtime is not None
        assert list(discovered_tools) == ["healthy"]
        assert TEST_TOOL_NAME in [tool.tool_name for tool in discovered_tools["healthy"]]
        assert "mcp__healthy__get_live_test_token" in resolved_by_name
        assert len(merged_tools) == 1
        assert "server 'local'" in caplog.text
        assert "skipping server 'local'" in caplog.text


def test_prepare_mcp_tools_proceeds_when_all_mcp_servers_are_unavailable(caplog, monkeypatch):
    failing_config = MCPServerConfig(name="local", url="http://127.0.0.1:1/mcp")

    async def fake_list_tools(self, server):
        raise MCPRuntimeError(
            f"Unable to connect to MCP server '{server.name}' at {server.url}: connect failed"
        )

    monkeypatch.setattr("gallama.remote_mcp.orchestrator.MCPRuntime.list_tools", fake_list_tools)
    caplog.set_level(logging.WARNING)

    async def _run():
        return await _prepare_mcp_tools(
            base_query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ),
            mcp_servers=[failing_config],
        )

    runtime, discovered_tools, resolved_by_name, merged_tools = asyncio.run(_run())

    assert runtime is not None
    assert discovered_tools == {}
    assert resolved_by_name == {}
    assert merged_tools == []
    assert "All configured MCP servers were unavailable" in caplog.text
