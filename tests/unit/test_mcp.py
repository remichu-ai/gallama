import asyncio
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest


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
SRC_TESTS_DIR = os.path.join(ROOT_DIR, "src", "tests")
if SRC_TESTS_DIR not in sys.path:
    sys.path.insert(0, SRC_TESTS_DIR)

from helpers.dummy_mcp_server import TEST_TOKEN, TEST_TOOL_NAME, run_dummy_mcp_server
from gallama.api_response.api_formatter import ResponsesFormatter
from gallama.api_response.chat_response import chat_completion_response_stream
from gallama.data_classes.data_class import AnthropicMessagesRequest, ChatMLQuery
from gallama.data_classes.data_class import TagDefinition
from gallama.data_classes.generation_data_class import GenEnd, GenQueue, GenQueueDynamic, GenStart, GenText, GenerationStats
from gallama.data_classes.responses_api import ResponseMCPListToolsItem, ResponsesCreateRequest
from gallama.remote_mcp.models import MCPResolvedTool
from gallama.remote_mcp.models import MCPServerConfig
from gallama.remote_mcp.orchestrator import MCPStreamController
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


def test_responses_request_tolerates_hosted_builtin_tools():
    request = ResponsesCreateRequest.model_validate(
        {
            "model": "test-model",
            "input": "Hello",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup_weather",
                    "description": "Lookup weather",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "type": "web_search",
                    "external_web_access": False,
                },
            ],
        }
    )

    chat_query = request.to_chat_ml_query()

    assert chat_query.tools is not None
    assert len(chat_query.tools) == 1
    assert chat_query.tools[0].function.name == "lookup_weather"


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


def test_chat_completion_stream_can_continue_after_suppressed_tool_calls():
    synthetic_tool_name = "mcp__demo__get_weather"

    def tool_post_processor(_text: str, _extra_vars: dict | None = None):
        return [
            {
                "id": "call_1",
                "type": "function",
                "index": 0,
                "function": {
                    "name": synthetic_tool_name,
                    "arguments": "{\"city\": \"Singapore\"}",
                },
            }
        ]

    async def _run():
        tool_tag = TagDefinition(
            tag_type="tool_calls",
            api_tag="tool_calls",
            role="assistant",
            post_processor=tool_post_processor,
            wait_till_complete=True,
        )
        gen_queue = GenQueueDynamic()

        gen_queue.put_nowait(GenStart(gen_type=tool_tag))
        gen_queue.put_nowait(GenText(content='{"name":"get_weather"}'))
        gen_queue.put_nowait(GenerationStats(stop_reason="tool_use"))
        gen_queue.put_nowait(GenEnd())

        intercepted_calls = []
        continued = False

        async def intercept_tool_calls(tool_calls):
            intercepted_calls.extend(tool_calls)
            return True

        async def continue_after_turn():
            nonlocal continued
            if continued:
                return False

            continued = True
            gen_queue.swap(GenQueue())
            gen_queue.put_nowait(GenText(content="Final weather answer"))
            gen_queue.put_nowait(
                GenerationStats(
                    stop_reason="end_turn",
                    input_tokens_count=5,
                    output_tokens_count=3,
                )
            )
            gen_queue.put_nowait(GenEnd())
            return True

        events = []
        async for event in chat_completion_response_stream(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="openai",
            tag_definitions=[tool_tag],
            tool_calls_interceptor=intercept_tool_calls,
            turn_end_interceptor=continue_after_turn,
        ):
            events.append(event)

        payloads = [event["data"] for event in events]
        streamed_content = ""
        for payload in payloads:
            if payload == "[DONE]":
                continue
            data = json.loads(payload)
            for choice in data.get("choices", []):
                streamed_content += choice.get("delta", {}).get("content", "")

        assert intercepted_calls
        assert intercepted_calls[0]["function"]["name"] == synthetic_tool_name
        assert streamed_content == "Final weather answer"
        assert all(synthetic_tool_name not in payload for payload in payloads)
        assert payloads[-1] == "[DONE]"

    asyncio.run(_run())


def test_chat_completion_stream_requests_stop_after_invalid_tool_continuation():
    def tool_post_processor(_text: str, _extra_vars: dict | None = None):
        return [
            {
                "id": "call_1",
                "type": "function",
                "index": 0,
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Singapore\"}",
                },
            }
        ]

    async def _run():
        tool_tag = TagDefinition(
            start_marker="<tool>",
            end_marker="</tool>",
            include_markers=True,
            tag_type="tool_calls",
            api_tag="tool_calls",
            role="assistant",
            post_processor=tool_post_processor,
            wait_till_complete=True,
            allowed_next_tag=[],
        )
        gen_queue = GenQueueDynamic()
        gen_queue.put_nowait(
            GenText(content="<tool>{\"name\":\"get_weather\"}</tool>\n\nhallucinated answer")
        )
        gen_queue.put_nowait(
            GenerationStats(
                stop_reason="tool_use",
                input_tokens_count=5,
                output_tokens_count=3,
            )
        )
        gen_queue.put_nowait(GenEnd())

        stop_requested = False
        events = []

        def request_stop():
            nonlocal stop_requested
            stop_requested = True

        async for event in chat_completion_response_stream(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="openai",
            tag_definitions=[tool_tag],
            generation_stop_callback=request_stop,
        ):
            events.append(event)

        payloads = [event["data"] for event in events]
        assert stop_requested is True
        assert all("hallucinated answer" not in payload for payload in payloads)

    asyncio.run(_run())


def test_responses_stream_can_emit_mcp_trace_items_before_final_text():
    synthetic_tool_name = "mcp__demo__get_weather"

    def tool_post_processor(_text: str, _extra_vars: dict | None = None):
        return [
            {
                "id": "call_1",
                "type": "function",
                "index": 0,
                "function": {
                    "name": synthetic_tool_name,
                    "arguments": "{\"city\": \"Singapore\"}",
                },
            }
        ]

    async def _run():
        tool_tag = TagDefinition(
            tag_type="tool_calls",
            api_tag="tool_calls",
            role="assistant",
            post_processor=tool_post_processor,
            wait_till_complete=True,
        )
        gen_queue = GenQueueDynamic()

        gen_queue.put_nowait(GenStart(gen_type=tool_tag))
        gen_queue.put_nowait(GenText(content='{"name":"get_weather"}'))
        gen_queue.put_nowait(GenerationStats(stop_reason="tool_use"))
        gen_queue.put_nowait(GenEnd())

        request_model = ResponsesCreateRequest.model_validate(
            {
                "model": "test-model",
                "input": "Weather?",
                "stream": True,
            }
        )

        formatter_holder = {}
        pending_events = []
        continued = False

        def formatter_ready(formatter):
            assert isinstance(formatter, ResponsesFormatter)
            formatter_holder["formatter"] = formatter
            pending_events.extend(
                formatter.append_output_items(
                    [
                        ResponseMCPListToolsItem(
                            server_label="demo",
                            tools=[
                                {
                                    "name": "get_weather",
                                    "description": "Get weather",
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"],
                                    },
                                }
                            ],
                        )
                    ]
                )
            )

        def drain_pending_events():
            events = list(pending_events)
            pending_events.clear()
            return events

        async def intercept_tool_calls(_tool_calls):
            return True

        async def continue_after_turn():
            nonlocal continued
            if continued:
                return False

            continued = True
            formatter = formatter_holder["formatter"]
            pending_events.extend(
                formatter.append_mcp_call_started(
                    call_id="mcp_call_1",
                    arguments='{"city":"Singapore"}',
                    name="get_weather",
                    server_label="demo",
                )
            )
            await asyncio.sleep(0.02)
            pending_events.extend(
                formatter.append_mcp_call_completed(
                    call_id="mcp_call_1",
                    output="Sunny, 31C",
                    error=None,
                )
            )

            gen_queue.swap(GenQueue())
            gen_queue.put_nowait(GenText(content="Final weather answer"))
            gen_queue.put_nowait(
                GenerationStats(
                    stop_reason="end_turn",
                    input_tokens_count=5,
                    output_tokens_count=3,
                    total_tokens_count=8,
                )
            )
            gen_queue.put_nowait(GenEnd())
            return True

        events = []
        async for event in chat_completion_response_stream(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="responses",
            tag_definitions=[tool_tag],
            formatter_kwargs={"request_model": request_model},
            formatter_ready_callback=formatter_ready,
            extra_events_getter=drain_pending_events,
            tool_calls_interceptor=intercept_tool_calls,
            turn_end_interceptor=continue_after_turn,
        ):
            events.append(event)

        decoded_events = [
            {"event": event.get("event"), "data": json.loads(event["data"])}
            for event in events
        ]
        added_item_types = [
            payload["data"]["item"]["type"]
            for payload in decoded_events
            if payload["event"] == "response.output_item.added"
        ]
        text_deltas = [
            payload["data"]["delta"]
            for payload in decoded_events
            if payload["event"] == "response.output_text.delta"
        ]
        event_names = [payload["event"] for payload in decoded_events]
        completed_payload = next(
            payload["data"]["response"]
            for payload in decoded_events
            if payload["event"] == "response.completed"
        )

        assert added_item_types[:2] == ["mcp_list_tools", "mcp_call"]
        assert "response.mcp_call_arguments.done" in event_names
        assert "response.mcp_call.in_progress" in event_names
        assert "response.mcp_call.completed" in event_names
        assert "".join(text_deltas) == "Final weather answer"
        assert [item["type"] for item in completed_payload["output"]] == [
            "mcp_list_tools",
            "mcp_call",
            "message",
        ]
        assert completed_payload["output"][1]["status"] == "completed"

    asyncio.run(_run())


def test_anthropic_stream_can_emit_mcp_blocks_before_final_text():
    synthetic_tool_name = "mcp__demo__get_weather"

    def tool_post_processor(_text: str, _extra_vars: dict | None = None):
        return [
            {
                "id": "call_1",
                "type": "function",
                "index": 0,
                "function": {
                    "name": synthetic_tool_name,
                    "arguments": "{\"city\": \"Singapore\"}",
                },
            }
        ]

    async def _run():
        tool_tag = TagDefinition(
            tag_type="tool_calls",
            api_tag="tool_calls",
            role="assistant",
            post_processor=tool_post_processor,
            wait_till_complete=True,
        )
        gen_queue = GenQueueDynamic()

        gen_queue.put_nowait(GenStart(gen_type=tool_tag))
        gen_queue.put_nowait(GenText(content='{"name":"get_weather"}'))
        gen_queue.put_nowait(GenerationStats(stop_reason="tool_use"))
        gen_queue.put_nowait(GenEnd())

        formatter_holder = {}
        pending_events = []
        continued = False

        def formatter_ready(formatter):
            formatter_holder["formatter"] = formatter

        def drain_pending_events():
            events = list(pending_events)
            pending_events.clear()
            return events

        async def intercept_tool_calls(_tool_calls):
            return True

        async def continue_after_turn():
            nonlocal continued
            if continued:
                return False

            continued = True
            formatter = formatter_holder["formatter"]
            pending_events.extend(
                formatter.append_mcp_tool_use_block(
                    call_id="call_1",
                    name="get_weather",
                    server_name="demo",
                    arguments={"city": "Singapore"},
                )
            )
            await asyncio.sleep(0.02)
            pending_events.extend(
                formatter.append_mcp_tool_result_block(
                    tool_use_id="call_1",
                    output_text="Sunny, 31C",
                    is_error=False,
                )
            )

            gen_queue.swap(GenQueue())
            gen_queue.put_nowait(GenText(content="Final weather answer"))
            gen_queue.put_nowait(
                GenerationStats(
                    stop_reason="end_turn",
                    input_tokens_count=5,
                    output_tokens_count=3,
                    total_tokens_count=8,
                )
            )
            gen_queue.put_nowait(GenEnd())
            return True

        events = []
        async for event in chat_completion_response_stream(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="anthropic",
            tag_definitions=[tool_tag],
            formatter_ready_callback=formatter_ready,
            extra_events_getter=drain_pending_events,
            tool_calls_interceptor=intercept_tool_calls,
            turn_end_interceptor=continue_after_turn,
        ):
            events.append(event)

        decoded_events = [
            {
                "event": event.get("event"),
                "data": json.loads(event["data"]),
            }
            for event in events
        ]
        started_blocks = [
            payload["data"]["content_block"]["type"]
            for payload in decoded_events
            if payload["event"] == "content_block_start"
            and payload["data"].get("content_block")
        ]
        text_deltas = [
            payload["data"]["delta"]["text"]
            for payload in decoded_events
            if payload["event"] == "content_block_delta"
            and payload["data"].get("delta", {}).get("type") == "text_delta"
        ]

        assert started_blocks[:2] == ["mcp_tool_use", "mcp_tool_result"]
        assert "".join(text_deltas) == "Final weather answer"
        assert decoded_events[-1]["event"] == "message_stop"

    asyncio.run(_run())


def test_mcp_stream_controller_only_suppresses_full_mcp_tool_turns():
    async def _chat_stub(**_kwargs):
        return None

    controller = MCPStreamController(
        provider="openai",
        base_query=ChatMLQuery.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ),
        llm=SimpleNamespace(
            model_name="test-model",
            prompt_eng=SimpleNamespace(tag_definitions=[]),
            chat=_chat_stub,
        ),
        request=None,
        conversation_messages=[],
        mcp_servers=[],
    )

    controller.resolved_by_synthetic_name = {
        "mcp__demo__get_weather": (
            MCPServerConfig(name="demo", url="http://localhost:9000/mcp"),
            MCPResolvedTool(
                server_name="demo",
                tool_name="get_weather",
                synthetic_name="mcp__demo__get_weather",
            ),
        )
    }

    async def _run():
        assert await controller.intercept_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mcp__demo__get_weather",
                        "arguments": "{\"city\":\"Singapore\"}",
                    },
                }
            ]
        )

        controller.current_turn_has_suppressed_mcp = False
        assert not await controller.intercept_tool_calls(
            [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": "{\"city\":\"Singapore\"}",
                    },
                }
            ]
        )

        with pytest.raises(Exception):
            await controller.intercept_tool_calls(
                [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "mcp__demo__get_weather",
                            "arguments": "{\"city\":\"Singapore\"}",
                        },
                    },
                    {
                        "id": "call_4",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Singapore\"}",
                        },
                    },
                ]
            )

    asyncio.run(_run())
