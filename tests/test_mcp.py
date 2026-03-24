import asyncio
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

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_TESTS_DIR = os.path.join(ROOT_DIR, "src", "tests")
if SRC_TESTS_DIR not in sys.path:
    sys.path.insert(0, SRC_TESTS_DIR)

from dummy_mcp_server import TEST_TOKEN, TEST_TOOL_NAME, run_dummy_mcp_server
from gallama.data_classes.data_class import AnthropicMessagesRequest, ChatMLQuery
from gallama.data_classes.responses_api import ResponsesCreateRequest
from gallama.remote_mcp.models import MCPServerConfig
from gallama.remote_mcp.runtime import MCPRuntime


def test_chat_query_splits_mcp_tools_from_function_tools():
    query = ChatMLQuery.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Lookup weather",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                },
                {
                    "type": "mcp",
                    "server_label": "demo",
                    "server_url": "http://localhost:9000/mcp",
                    "allowed_tools": [TEST_TOOL_NAME],
                },
            ],
        }
    )

    function_tools, mcp_servers = query.split_tools()

    assert len(function_tools) == 1
    assert function_tools[0].function.name == "lookup_weather"
    assert len(mcp_servers) == 1
    assert mcp_servers[0].name == "demo"
    assert mcp_servers[0].allowed_tools == [TEST_TOOL_NAME]


def test_responses_request_extracts_mcp_servers():
    request = ResponsesCreateRequest.model_validate(
        {
            "model": "test-model",
            "input": "Hello",
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "demo",
                    "server_url": "http://localhost:9000/mcp",
                    "authorization_token": "secret-token",
                }
            ],
        }
    )

    servers = request.get_mcp_server_configs()

    assert len(servers) == 1
    assert servers[0].name == "demo"
    assert servers[0].authorization_token == "secret-token"
    assert request.to_chat_ml_query().tools is None


def test_anthropic_request_extracts_mcp_servers():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Hello"}],
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "demo-mcp",
                    "url": "http://localhost:9000/mcp",
                    "authorization_token": "abc123",
                }
            ],
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "demo-mcp",
                    "allowed_tools": [TEST_TOOL_NAME],
                }
            ],
        }
    )

    servers = request.get_mcp_server_configs()
    chat_query = request.get_ChatMLQuery()

    assert len(servers) == 1
    assert servers[0].name == "demo-mcp"
    assert servers[0].allowed_tools == [TEST_TOOL_NAME]
    assert chat_query.tools is None


def test_mcp_runtime_lists_and_calls_tools_over_streamable_http():
    with run_dummy_mcp_server() as server:
        runtime = MCPRuntime()
        config = MCPServerConfig(name="demo", url=server.url)

        async def _run():
            tools = await runtime.list_tools(config)
            assert len(tools) == 1
            assert tools[0].tool_name == TEST_TOOL_NAME

            trace = await runtime.call_tool(
                server=config,
                tool=tools[0],
                arguments={},
                call_id="call_123",
            )
            assert trace.output_text is not None
            assert TEST_TOKEN in trace.output_text
            assert trace.error is None
            assert server.count_method("tools/list") >= 1
            assert server.count_method("tools/call") >= 1

        asyncio.run(_run())
