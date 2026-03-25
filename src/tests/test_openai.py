"""
Comprehensive test suite for a local OpenAI-compatible Chat Completions endpoint.

Every API response (and streaming chunks) is logged to a JSON file so you can
inspect exactly what the server returned for any failing test.

Usage:
    LOCAL_BASE_URL=http://localhost:8000/v1 LOCAL_API_KEY=test python test_openai_endpoint.py

    # Custom log path / model:
    TEST_LOG_FILE=debug.json TEST_MODEL=my-model python test_openai_endpoint.py
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
from typing import Any

import openai
from dummy_mcp_server import (
    TEST_TOKEN,
    TEST_USER_PROMPT,
    build_openai_mcp_tool,
    run_dummy_mcp_server,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = os.getenv("TEST_MODEL", "gpt-4o")
LOG_FILE = os.getenv("TEST_LOG_FILE", "test_responses.json")
COMMON_SYSTEM_PROMPT = os.getenv(
    "TEST_SYSTEM_PROMPT",
    "You are a helpful assistant. Follow the user's instructions exactly.",
)
PIRATE_SYSTEM_PROMPT = (
    f"{COMMON_SYSTEM_PROMPT}\n\nYou are a pirate. Every answer must contain 'Arrr'."
)
JSON_MODE_SYSTEM_PROMPT = (
    f"{COMMON_SYSTEM_PROMPT}\n\nYou always respond in JSON format."
)
JSON_SCHEMA_SYSTEM_PROMPT = (
    f"{COMMON_SYSTEM_PROMPT}\n\nYou always respond in the requested JSON format."
)
VISION_JSON_SYSTEM_PROMPT = (
    f"{COMMON_SYSTEM_PROMPT}\n\nYou analyze images and respond in the requested JSON format."
)

# Path to a local cat image used for vision tests
LOCAL_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "cat1.jpg")

# Simple tools reused across tests (OpenAI function-calling format)
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'",
                }
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2'",
                }
            },
            "required": ["expression"],
        },
    },
}

STOCK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'AAPL'",
                }
            },
            "required": ["ticker"],
        },
    },
}

# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def _load_image_data_url(path: str) -> str:
    """Load a local image file and return a data URL (data:image/...;base64,...)."""
    ext = os.path.splitext(path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return f"data:{media_type};base64,{data}"


# ---------------------------------------------------------------------------
# Response logger — collects every API response for post-mortem debugging
# ---------------------------------------------------------------------------
_log_entries: list[dict[str, Any]] = []


def _resp_to_dict(resp) -> dict[str, Any] | None:
    """Serialize an OpenAI response object to a plain dict."""
    if resp is None:
        return None
    # Already a dict (e.g. from multi-step tests)
    if isinstance(resp, dict):
        return resp
    try:
        return resp.model_dump(mode="json")
    except Exception:
        pass
    try:
        return resp.to_dict()
    except Exception:
        pass
    return {"__repr__": repr(resp)}


def _log_response(
    test_name: str,
    *,
    request_params: dict[str, Any] | None = None,
    response: Any = None,
    stream_chunks: list[Any] | None = None,
    error: Exception | None = None,
    passed: bool,
    detail: str = "",
):
    """Append a structured entry to the in-memory log."""
    entry: dict[str, Any] = {
        "test": test_name,
        "passed": passed,
        "detail": detail,
        "timestamp": time.time(),
    }
    if request_params:
        entry["request"] = request_params
    if response is not None:
        entry["response"] = _resp_to_dict(response)
    if stream_chunks is not None:
        entry["stream_chunks"] = [_resp_to_dict(c) for c in stream_chunks]
    if error is not None:
        entry["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        }
        # If it's an API error, capture the raw HTTP response body
        if hasattr(error, "response") and error.response is not None:
            try:
                entry["error"]["http_status"] = error.response.status_code
                entry["error"]["response_body"] = error.response.text
            except Exception:
                pass
        if hasattr(error, "body"):
            entry["error"]["error_body"] = error.body
    _log_entries.append(entry)


def _save_log():
    """Write accumulated log entries to disk."""
    path = os.path.abspath(LOG_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "total_tests": len(_log_entries),
                "passed": sum(1 for e in _log_entries if e["passed"]),
                "failed": sum(1 for e in _log_entries if not e["passed"]),
                "entries": _log_entries,
            },
            f,
            indent=2,
            default=str,
        )
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass = 0
_fail = 0
_skip = 0


def _report(name: str, passed: bool, detail: str = ""):
    global _pass, _fail
    icon = "✅" if passed else "❌"
    if passed:
        _pass += 1
    else:
        _fail += 1
    print(f"  {icon}  {name}" + (f"  — {detail}" if detail else ""))


def _skip_test(name: str, reason: str):
    global _skip
    _skip += 1
    print(f"  ⏭️  {name}  — SKIPPED: {reason}")


def _section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _get_text(resp) -> str:
    """Extract the assistant's text content from the response."""
    msg = resp.choices[0].message
    return msg.content or ""


def _get_tool_calls(resp) -> list:
    """Extract tool calls from the response."""
    msg = resp.choices[0].message
    return msg.tool_calls or []


# ---------------------------------------------------------------------------
# 1. Basic message creation (non-streaming)
# ---------------------------------------------------------------------------
def test_basic_message(client: openai.OpenAI):
    """Simple single-turn user message -> assistant response."""
    name = "Basic message"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert resp.id, "Response missing id"
        assert resp.object == "chat.completion", f"Unexpected object: {resp.object}"
        choice = resp.choices[0]
        assert choice.message.role == "assistant", f"Unexpected role: {choice.message.role}"
        assert choice.message.content, "Empty content"
        assert choice.finish_reason == "stop", f"Unexpected finish_reason: {choice.finish_reason}"
        assert resp.usage.prompt_tokens > 0, "prompt_tokens should be > 0"
        assert resp.usage.completion_tokens > 0, "completion_tokens should be > 0"
        text = _get_text(resp)
        _report(name, True, f"Got: {text!r}")
        _log_response(name, request_params=params, response=resp, passed=True, detail=f"Got: {text!r}")
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 2. System prompt
# ---------------------------------------------------------------------------
def test_system_prompt(client: openai.OpenAI):
    """Ensure system prompt influences the response."""
    name = "System prompt"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
            {"role": "user", "content": "How are you today?"},
        ],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp).lower()
        assert "arrr" in text, f"Expected 'arrr' in response, got: {text}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 3. Multi-turn conversation
# ---------------------------------------------------------------------------
def test_multi_turn(client: openai.OpenAI):
    """Multi-turn conversation preserves context."""
    name = "Multi-turn conversation"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        messages=[
            {"role": "user", "content": "My name is Zephyr."},
            {"role": "assistant", "content": "Nice to meet you, Zephyr!"},
            {"role": "user", "content": "What is my name?"},
        ],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp)
        assert "Zephyr" in text, f"Expected 'Zephyr' in response: {text}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 4. Streaming (basic)
# ---------------------------------------------------------------------------
def test_streaming_basic(client: openai.OpenAI):
    """Stream a response and verify chunks arrive."""
    name = "Streaming (basic)"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )
    chunks = []
    try:
        full_text = ""
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            chunks.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_text += delta.content

        assert len(full_text) > 0, "No text received"
        detail = f"Chunks: {len(chunks)}, text length: {len(full_text)}"
        _report(name, True, detail)
        _log_response(name, request_params=params, stream_chunks=chunks, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, stream_chunks=chunks if chunks else None, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 5. Streaming -- finish_reason propagation
# ---------------------------------------------------------------------------
def test_streaming_finish_reason(client: openai.OpenAI):
    """Verify finish_reason is present in the final chunk."""
    name = "Streaming (finish_reason)"
    params = dict(
        model=MODEL,
        max_tokens=6000,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )
    chunks = []
    try:
        finish_reason = None
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "stop", f"Expected 'stop', got: {finish_reason}"
        _report(name, True, f"finish_reason={finish_reason}")
        _log_response(name, request_params=params, stream_chunks=chunks, passed=True, detail=f"finish_reason={finish_reason}")
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, stream_chunks=chunks if chunks else None, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 6. Stop sequences
# ---------------------------------------------------------------------------
def test_stop_sequences(client: openai.OpenAI):
    """Stop generation when a custom stop sequence is hit."""
    name = "Stop sequences"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        stop=["STOP"],
        messages=[{"role": "user", "content": "Write the sentence 'I will now STOP talking' verbatim."}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert resp.choices[0].finish_reason == "stop", f"Expected stop, got: {resp.choices[0].finish_reason}"
        text = _get_text(resp)
        assert "STOP" not in text, f"Stop sequence should have been removed from output: {text}"
        _report(name, True, f"finish_reason={resp.choices[0].finish_reason}")
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 7. Max tokens limit
# ---------------------------------------------------------------------------
def test_max_tokens(client: openai.OpenAI):
    """Ensure max_tokens is respected and finish_reason is length."""
    name = "Max tokens limit"
    params = dict(
        model=MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": "Write a very long essay about the history of the universe."}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert resp.choices[0].finish_reason == "length", f"Expected length, got: {resp.choices[0].finish_reason}"
        assert resp.usage.completion_tokens <= 10, f"completion_tokens ({resp.usage.completion_tokens}) exceeded expected cap"
        _report(name, True, f"completion_tokens={resp.usage.completion_tokens}")
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 8. Temperature
# ---------------------------------------------------------------------------
def test_temperature(client: openai.OpenAI):
    """Temperature 0.7 should yield deterministic (or near-deterministic) output."""
    name = "Temperature 0.7 determinism"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        temperature=0.7,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    )
    r1 = r2 = None
    try:
        r1 = client.chat.completions.create(**params)
        r2 = client.chat.completions.create(**params)
        t1 = _get_text(r1).strip()
        t2 = _get_text(r2).strip()
        same = t1 == t2
        detail = f"r1={t1!r}, r2={t2!r} -- {'match' if same else 'MISMATCH (may be acceptable)'}"
        _report(name, same, detail)
        _log_response(name, request_params=params, response={"r1": _resp_to_dict(r1), "r2": _resp_to_dict(r2)}, passed=same, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response={"r1": _resp_to_dict(r1), "r2": _resp_to_dict(r2)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 9. Single tool use
# ---------------------------------------------------------------------------
def test_single_tool_use(client: openai.OpenAI):
    """Model should invoke a single tool when appropriate."""
    name = "Single tool use"
    params = dict(
        model=MODEL,
        max_tokens=3000,
        tools=[WEATHER_TOOL],
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        tool_calls = _get_tool_calls(resp)
        assert len(tool_calls) >= 1, "No tool calls found"
        tc = tool_calls[0]
        assert tc.function.name == "get_weather", f"Wrong tool: {tc.function.name}"
        args = json.loads(tc.function.arguments)
        assert resp.choices[0].finish_reason == "tool_calls", f"finish_reason={resp.choices[0].finish_reason}"
        detail = f"tool={tc.function.name}, args={args}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 10. Tool use -> tool_result round-trip
# ---------------------------------------------------------------------------
def test_tool_result_roundtrip(client: openai.OpenAI):
    """Full cycle: model calls tool -> we return result -> model responds."""
    name = "Tool result round-trip"
    resp1 = resp2 = None
    try:
        params1 = dict(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        )
        resp1 = client.chat.completions.create(**params1)
        tool_calls = _get_tool_calls(resp1)
        assert tool_calls, "Model did not call tool"
        tc = tool_calls[0]

        resp2 = client.chat.completions.create(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
                resp1.choices[0].message,
                {"role": "tool", "tool_call_id": tc.id, "content": "Sunny, 24C, light breeze"},
            ],
        )
        text = _get_text(resp2)
        assert resp2.choices[0].finish_reason == "stop", f"finish_reason={resp2.choices[0].finish_reason}"
        assert len(text) > 0, "Empty final response"
        _report(name, True, f"Final response contains weather info: {'24' in text or 'sunny' in text.lower()}")
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 10b. MCP tool use against a live dummy MCP server
# ---------------------------------------------------------------------------
def test_mcp_tool_live_server(client: openai.OpenAI):
    """Use a real MCP server over HTTP and verify Gallama executes it server-side."""
    name = "MCP tool (live server)"
    resp = None
    params = None
    mcp_requests: list[dict[str, Any]] = []
    try:
        with run_dummy_mcp_server() as mcp_server:
            params = dict(
                model=MODEL,
                max_tokens=3000,
                tools=[build_openai_mcp_tool(mcp_server.url)],
                messages=[{"role": "user", "content": TEST_USER_PROMPT}],
            )
            resp = client.chat.completions.create(**params)
            mcp_requests = mcp_server.requests

            text = _get_text(resp).strip()
            assert TEST_TOKEN in text, f"Expected token {TEST_TOKEN!r} in response: {text!r}"
            assert mcp_server.count_method("tools/list") >= 1, "MCP server did not receive tools/list"
            assert mcp_server.count_method("tools/call") >= 1, "MCP server did not receive tools/call"

            detail = (
                f"token returned, tools/list={mcp_server.count_method('tools/list')}, "
                f"tools/call={mcp_server.count_method('tools/call')}"
            )
            _report(name, True, detail)
            _log_response(
                name,
                request_params=params,
                response={"chat": _resp_to_dict(resp), "mcp_requests": mcp_requests},
                passed=True,
                detail=detail,
            )
    except Exception as e:
        _report(name, False, str(e))
        _log_response(
            name,
            request_params=params,
            response={"chat": _resp_to_dict(resp), "mcp_requests": mcp_requests},
            error=e,
            passed=False,
            detail=str(e),
        )


# ---------------------------------------------------------------------------
# 11. Multiple / parallel tool use
# ---------------------------------------------------------------------------
def test_parallel_tool_use(client: openai.OpenAI):
    """Model should invoke multiple tools in a single turn."""
    name = "Parallel tool use"
    params = dict(
        model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL, STOCK_TOOL],
        messages=[{"role": "user", "content": "I need two things at once, provide me both in 1 turn: 1) the weather in London and 2) the stock price of AAPL. Please call both tools."}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        # print(resp)
        tool_calls = _get_tool_calls(resp)
        tool_names = {tc.function.name for tc in tool_calls}
        both = {"get_weather", "get_stock_price"} <= tool_names
        detail = f"Tool calls: {len(tool_calls)}, names: {tool_names}" + (" (both present)" if both else " (missing one)")
        passed = len(tool_calls) >= 2
        _report(name, passed, detail)
        _log_response(name, request_params=params, response=resp, passed=passed, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 12. Parallel tool use round-trip
# ---------------------------------------------------------------------------
def test_parallel_tool_roundtrip(client: openai.OpenAI):
    """Full cycle with multiple tool calls and results."""
    name = "Parallel tool round-trip"
    resp1 = resp2 = None
    try:
        resp1 = client.chat.completions.create(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL, STOCK_TOOL],
            messages=[{"role": "user", "content": "What's the weather in Paris and the stock price of MSFT?"}],
        )
        tool_calls = _get_tool_calls(resp1)
        assert len(tool_calls) >= 1, "No tool calls"

        fake_results = {"get_weather": "Cloudy, 18C in Paris", "get_stock_price": "MSFT is trading at $420.50"}
        tool_messages = [
            {"role": "tool", "tool_call_id": tc.id, "content": fake_results.get(tc.function.name, "Unknown tool")}
            for tc in tool_calls
        ]

        resp2 = client.chat.completions.create(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL, STOCK_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Paris and the stock price of MSFT?"},
                resp1.choices[0].message,
                *tool_messages,
            ],
        )
        text = _get_text(resp2)
        assert len(text) > 0, "Empty final text"
        _report(name, True, f"Final answer length: {len(text)} chars")
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 13. tool_choice: auto / required / specific
# ---------------------------------------------------------------------------
def test_tool_choice_auto(client: openai.OpenAI):
    """tool_choice='auto' lets the model decide."""
    name = "tool_choice=auto"
    params = dict(
        model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL], tool_choice="auto",
        messages=[{"role": "user", "content": "Tell me a joke."}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        tool_calls = _get_tool_calls(resp)
        detail = f"Tool calls: {len(tool_calls)} (expected 0)"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


def test_tool_choice_required(client: openai.OpenAI):
    """tool_choice='required' forces the model to use some tool."""
    name = "tool_choice=required"
    params = dict(
        model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL, CALCULATOR_TOOL], tool_choice="required",
        messages=[{"role": "user", "content": "Hello!"}],
        # messages=[{"role": "user", "content": "What the weather in Ha NOi"}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        tool_calls = _get_tool_calls(resp)
        assert len(tool_calls) >= 1, "Expected at least one tool call with tool_choice=required"
        detail = f"Forced tool: {tool_calls[0].function.name}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


def test_tool_choice_specific(client: openai.OpenAI):
    """tool_choice forces a specific function."""
    name = "tool_choice=specific tool"
    params = dict(
        model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL, CALCULATOR_TOOL],
        tool_choice={"type": "function", "function": {"name": "calculator"}},
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        tool_calls = _get_tool_calls(resp)
        assert len(tool_calls) >= 1, "No tool call"
        assert tool_calls[0].function.name == "calculator", f"Expected calculator, got {tool_calls[0].function.name}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 14. Streaming + tool use
# ---------------------------------------------------------------------------
def test_streaming_tool_use(client: openai.OpenAI):
    """Stream a response that includes a tool call."""
    name = "Streaming + tool use"
    params = dict(
        model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL],
        messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
        stream=True,
    )
    chunks = []
    try:
        tool_calls_by_index: dict[int, dict] = {}
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            chunks.append(chunk)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_by_index[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_by_index[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_by_index[idx]["arguments"] += tc_delta.function.arguments

        assert tool_calls_by_index, "No tool calls in streamed response"
        first = list(tool_calls_by_index.values())[0]
        detail = f"Streamed tool: {first['name']}"
        _report(name, True, detail)
        _log_response(name, request_params=params, stream_chunks=chunks, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, stream_chunks=chunks if chunks else None, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 15. Structured output / JSON mode
# ---------------------------------------------------------------------------
def test_json_mode(client: openai.OpenAI):
    """response_format=json_object should return valid JSON."""
    name = "JSON mode"
    params = dict(
        model=MODEL, max_tokens=3000, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JSON_MODE_SYSTEM_PROMPT},
            {"role": "user", "content": "Give me a JSON object with keys 'name' and 'age' for a fictional character."},
        ],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp)
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        detail = f"Parsed JSON: {parsed}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 15b. Structured output / JSON Schema mode
# ---------------------------------------------------------------------------
def test_json_schema(client: openai.OpenAI):
    """response_format=json_schema should return JSON conforming to the given schema."""
    name = "JSON schema (structured output)"
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age", "hobbies"],
        "additionalProperties": False,
    }
    params = dict(
        model=MODEL,
        max_tokens=3000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fictional_character",
                "strict": True,
                "schema": schema,
            },
        },
        messages=[
            {
                "role": "system",
                "content": JSON_SCHEMA_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Give me a fictional character with a name, age (integer), and a list of hobbies.",
            },
        ],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp)
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        # Validate required keys exist
        for key in ("name", "age", "hobbies"):
            assert key in parsed, f"Missing required key: {key}"
        # Validate types
        assert isinstance(parsed["name"], str), f"name should be str, got {type(parsed['name'])}"
        assert isinstance(parsed["age"], int), f"age should be int, got {type(parsed['age'])}"
        assert isinstance(parsed["hobbies"], list), f"hobbies should be list, got {type(parsed['hobbies'])}"
        for i, h in enumerate(parsed["hobbies"]):
            assert isinstance(h, str), f"hobbies[{i}] should be str, got {type(h)}"
        # Validate no extra keys (additionalProperties: false)
        extra = set(parsed.keys()) - {"name", "age", "hobbies"}
        assert not extra, f"Unexpected extra keys: {extra}"
        detail = f"Parsed JSON: {parsed}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))

# ---------------------------------------------------------------------------
# 15c. Structured output / JSON Schema + Vision
# ---------------------------------------------------------------------------
def test_json_schema_vision(client: openai.OpenAI):
    """response_format=json_schema with a local image input should return conforming JSON."""
    name = "JSON schema + vision"
    schema = {
        "type": "object",
        "properties": {
            "animal": {"type": "string"},
            "color": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["animal", "color", "description"],
        "additionalProperties": False,
    }
    image_data_url = _load_image_data_url(LOCAL_IMAGE_PATH)
    params = dict(
        model=MODEL,
        max_tokens=3000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "image_analysis",
                "strict": True,
                "schema": schema,
            },
        },
        messages=[
            {
                "role": "system",
                "content": VISION_JSON_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Identify the animal in this image. Return its type, primary color, and a one-sentence description.",
                    },
                ],
            },
        ],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp)
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        for key in ("animal", "color", "description"):
            assert key in parsed, f"Missing required key: {key}"
        for key in ("animal", "color", "description"):
            assert isinstance(parsed[key], str), f"{key} should be str, got {type(parsed[key])}"
        extra = set(parsed.keys()) - {"animal", "color", "description"}
        assert not extra, f"Unexpected extra keys: {extra}"
        assert any(word in parsed["animal"].lower() for word in ("kitten", "cat")), \
            f"Expected 'kitten' or 'cat' in description, got: {parsed["animal"]}"

        detail = f"Parsed JSON: {parsed}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 16. Seed for reproducibility
# ---------------------------------------------------------------------------
def test_seed_determinism(client: openai.OpenAI):
    """Same seed + temperature 0.7 should yield identical output."""
    name = "Seed determinism"
    params = dict(
        model=MODEL, max_tokens=3000, temperature=0.7, seed=42,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    )
    r1 = r2 = None
    try:
        r1 = client.chat.completions.create(**params)
        r2 = client.chat.completions.create(**params)
        t1 = _get_text(r1).strip()
        t2 = _get_text(r2).strip()
        same = t1 == t2
        detail = f"r1={t1!r}, r2={t2!r} -- {'match' if same else 'MISMATCH'}"
        _report(name, same, detail)
        _log_response(name, request_params=params, response={"r1": _resp_to_dict(r1), "r2": _resp_to_dict(r2)}, passed=same, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response={"r1": _resp_to_dict(r1), "r2": _resp_to_dict(r2)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 17. Vision / image input
# ---------------------------------------------------------------------------
def test_vision_input(client: openai.OpenAI):
    """Send a local image of a cat and ask the model to describe it."""
    name = "Vision / image input (cat)"
    image_data_url = _load_image_data_url(LOCAL_IMAGE_PATH)
    params = dict(
        model=MODEL, max_tokens=3000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
                {"type": "text", "text": "What animal is in this image? Describe it briefly."},
            ],
        }],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp)

        assert len(text) > 0, "Empty response for image description"
        assert any(word in text.lower() for word in ("kitten", "cat")), \
            f"Expected 'kitten' or 'cat' in description, got: {text}"

        detail = f"Description: {text[:80]}..."
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 18. Top-p sampling
# ---------------------------------------------------------------------------
def test_top_p(client: openai.OpenAI):
    """Verify top_p parameter is accepted."""
    name = "top_p parameter"
    params = dict(model=MODEL, max_tokens=3000, top_p=0.9, messages=[{"role": "user", "content": "Say one word."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert _get_text(resp), "No output"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 19. Frequency & presence penalty
# ---------------------------------------------------------------------------
def test_frequency_penalty(client: openai.OpenAI):
    """Verify frequency_penalty parameter is accepted."""
    name = "frequency_penalty parameter"
    params = dict(model=MODEL, max_tokens=3000, frequency_penalty=0.5, messages=[{"role": "user", "content": "Say one word."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert _get_text(resp), "No output"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


def test_presence_penalty(client: openai.OpenAI):
    """Verify presence_penalty parameter is accepted."""
    name = "presence_penalty parameter"
    params = dict(model=MODEL, max_tokens=3000, presence_penalty=0.5, messages=[{"role": "user", "content": "Say one word."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert _get_text(resp), "No output"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 20. Multiple content blocks (mixed text)
# ---------------------------------------------------------------------------
def test_mixed_content_blocks(client: openai.OpenAI):
    """Multiple content blocks of different types in one user turn."""
    name = "Mixed content blocks"
    params = dict(
        model=MODEL, max_tokens=3000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "First, note that my favourite colour is blue."},
            {"type": "text", "text": "Now, what is my favourite colour?"},
        ]}],
    )
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        text = _get_text(resp).lower()
        assert "blue" in text, f"Expected 'blue' in: {text}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 21. Error handling -- invalid model
# ---------------------------------------------------------------------------
def test_error_invalid_model(client: openai.OpenAI):
    """Request with a nonsense model name should return an error."""
    name = "Error: invalid model"
    params = dict(model="nonexistent-model-xyz", max_tokens=3000, messages=[{"role": "user", "content": "Hi"}])
    try:
        resp = client.chat.completions.create(**params)
        _report(name, False, "Expected an error but got a 200")
        _log_response(name, request_params=params, response=resp, passed=False, detail="Expected error, got 200")
    except openai.NotFoundError as e:
        _report(name, True, "Got NotFoundError as expected")
        _log_response(name, request_params=params, error=e, passed=True, detail="NotFoundError")
    except openai.BadRequestError as e:
        _report(name, True, "Got BadRequestError as expected")
        _log_response(name, request_params=params, error=e, passed=True, detail="BadRequestError")
    except openai.APIError as e:
        _report(name, True, f"Got APIError: {e.status_code}")
        _log_response(name, request_params=params, error=e, passed=True, detail=f"APIError {e.status_code}")
    except Exception as e:
        _report(name, False, f"Unexpected error type: {type(e).__name__}: {e}")
        _log_response(name, request_params=params, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 22. Error handling -- empty messages
# ---------------------------------------------------------------------------
def test_error_empty_messages(client: openai.OpenAI):
    """Empty messages array should return a 400-level error."""
    name = "Error: empty messages"
    params = dict(model=MODEL, max_tokens=3000, messages=[])
    try:
        resp = client.chat.completions.create(**params)
        _report(name, False, "Expected an error but got a 200")
        _log_response(name, request_params=params, response=resp, passed=False, detail="Expected error, got 200")
    except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
        _report(name, True, "Got expected 4xx error")
        _log_response(name, request_params=params, error=e, passed=True, detail="Expected 4xx")
    except openai.APIError as e:
        _report(name, True, f"Got APIError: {e.status_code}")
        _log_response(name, request_params=params, error=e, passed=True, detail=f"APIError {e.status_code}")
    except Exception as e:
        _report(name, True, f"Client-side validation: {type(e).__name__}")
        _log_response(name, request_params=params, error=e, passed=True, detail=f"Client validation: {type(e).__name__}")


# ---------------------------------------------------------------------------
# 23. Usage / token counting
# ---------------------------------------------------------------------------
def test_usage_fields(client: openai.OpenAI):
    """Verify usage object has expected fields with sane values."""
    name = "Usage fields"
    params = dict(model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": "Hi."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        u = resp.usage
        assert u.prompt_tokens > 0, f"prompt_tokens={u.prompt_tokens}"
        assert u.completion_tokens > 0, f"completion_tokens={u.completion_tokens}"
        assert u.total_tokens == u.prompt_tokens + u.completion_tokens, (
            f"total_tokens mismatch: {u.total_tokens} != {u.prompt_tokens} + {u.completion_tokens}"
        )
        detail = f"prompt={u.prompt_tokens}, completion={u.completion_tokens}, total={u.total_tokens}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 24. Model field echo
# ---------------------------------------------------------------------------
def test_model_echo(client: openai.OpenAI):
    """Response should echo back the model identifier."""
    name = "Model echo"
    params = dict(model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": "Hi"}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        assert resp.model, "model field is empty"
        _report(name, True, f"model={resp.model}")
        _log_response(name, request_params=params, response=resp, passed=True, detail=f"model={resp.model}")
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 25. N > 1 (multiple choices)
# ---------------------------------------------------------------------------
def test_n_choices(client: openai.OpenAI):
    """Request n=2 completions and verify we get 2 choices back (Optional)."""
    name = "n=2 (multiple choices)"
    params = dict(model=MODEL, max_tokens=3000, n=2, messages=[{"role": "user", "content": "Pick a random color."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)

        # Make this optional: skip if the endpoint ignores n=2 and returns 1 choice
        if len(resp.choices) < 2:
            _skip_test(name, f"Expected 2 choices, got {len(resp.choices)} (Optional feature)")
            return

        t1 = resp.choices[0].message.content
        t2 = resp.choices[1].message.content
        detail = f"choice[0]={t1!r}, choice[1]={t2!r}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        # Also skip if the endpoint throws an explicit error for n > 1
        _skip_test(name, f"Optional feature error: {e}")
        _log_response(name, request_params=params, response=resp, error=e, passed=True, detail=f"Skipped: {e}")


# ---------------------------------------------------------------------------
# 26. Logprobs
# ---------------------------------------------------------------------------
def test_logprobs(client: openai.OpenAI):
    """Verify logprobs are returned when requested (Optional)."""
    name = "Logprobs"
    params = dict(model=MODEL, max_tokens=3000, logprobs=True, top_logprobs=3,
                  messages=[{"role": "user", "content": "Say hello."}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        lp = resp.choices[0].logprobs

        # Make this optional: skip if the endpoint ignores logprobs
        if lp is None:
            _skip_test(name, "logprobs is None (Optional feature)")
            return
        if not lp.content or len(lp.content) == 0:
            _skip_test(name, "No logprob content returned (Optional feature)")
            return

        first_token = lp.content[0]
        if first_token.top_logprobs is None:
            _skip_test(name, "No top_logprobs returned (Optional feature)")
            return

        detail = f"First token: {first_token.token!r}, logprob={first_token.logprob:.3f}, top_logprobs count={len(first_token.top_logprobs)}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as e:
        # Skip if the endpoint explicitly rejects the logprobs parameter
        _skip_test(name, f"Optional feature error: {e}")
        _log_response(name, request_params=params, response=resp, error=e, passed=True, detail=f"Skipped: {e}")


# ---------------------------------------------------------------------------
# 27. Tool error result handling
# ---------------------------------------------------------------------------
def test_tool_error_handling(client: openai.OpenAI):
    """Return an error-like tool result and verify model handles it gracefully."""
    name = "Tool error handling"
    resp1 = resp2 = None
    try:
        resp1 = client.chat.completions.create(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather in Atlantis?"}],
        )
        tool_calls = _get_tool_calls(resp1)
        if not tool_calls:
            _report(name, True, "Model answered without tool (acceptable)")
            _log_response(name, response=resp1, passed=True, detail="No tool call needed")
            return

        tc = tool_calls[0]
        resp2 = client.chat.completions.create(
            model=MODEL, max_tokens=3000, tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Atlantis?"},
                resp1.choices[0].message,
                {"role": "tool", "tool_call_id": tc.id, "content": "Error: Location not found. Atlantis is not a real place."},
            ],
        )
        text = _get_text(resp2)
        assert len(text) > 0, "Empty response after tool error"
        _report(name, True, f"Handled error gracefully, response length={len(text)}")
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 28. Sequential multi-tool (agentic loop)
# ---------------------------------------------------------------------------
def test_agentic_loop(client: openai.OpenAI):
    """Simulate an agentic loop: model calls tools sequentially until done."""
    name = "Agentic loop (multi-step)"
    all_responses = []
    try:
        tools = [WEATHER_TOOL, CALCULATOR_TOOL]
        messages = [{"role": "user", "content": "First get the weather in NYC, then use the calculator to compute the temperature in Fahrenheit if it's 22C (formula: C*9/5+32)."}]

        max_iterations = 5
        iteration = 0
        resp = None
        while iteration < max_iterations:
            iteration += 1
            resp = client.chat.completions.create(model=MODEL, max_tokens=3000, tools=tools, messages=messages)
            all_responses.append(resp)
            messages.append(resp.choices[0].message)

            tool_calls = _get_tool_calls(resp)
            if not tool_calls:
                break

            for tc in tool_calls:
                if tc.function.name == "get_weather":
                    result = "22C, partly cloudy in NYC"
                elif tc.function.name == "calculator":
                    result = "71.6"
                else:
                    result = "unknown"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        text = _get_text(resp) if resp else ""
        passed = iteration >= 2 and len(text) > 0
        detail = f"Completed in {iteration} iterations"
        _report(name, passed, detail)
        _log_response(name, response={f"step{i+1}": _resp_to_dict(r) for i, r in enumerate(all_responses)}, passed=passed, detail=detail)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, response={f"step{i+1}": _resp_to_dict(r) for i, r in enumerate(all_responses)}, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# 29. System fingerprint
# ---------------------------------------------------------------------------
def test_system_fingerprint(client: openai.OpenAI):
    """Verify system_fingerprint is present in response."""
    name = "System fingerprint"
    params = dict(model=MODEL, max_tokens=3000, messages=[{"role": "user", "content": "Hi"}])
    resp = None
    try:
        resp = client.chat.completions.create(**params)
        fp = resp.system_fingerprint
        _report(name, True, f"system_fingerprint={fp!r}" if fp else "system_fingerprint=None (optional)")
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as e:
        _report(name, False, str(e))
        _log_response(name, request_params=params, response=resp, error=e, passed=False, detail=str(e))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(client: openai.OpenAI):
    """Run every test and print a summary. All API responses saved to LOG_FILE."""
    global _pass, _fail, _skip
    _pass = _fail = _skip = 0
    _log_entries.clear()

    print("\n" + "=" * 60)
    print("  OpenAI Chat Completions API -- Compatibility Test Suite")
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {client.base_url}")
    print(f"  Response log: {os.path.abspath(LOG_FILE)}")
    print("=" * 60)

    _section("Basic Features")
    test_basic_message(client)
    test_system_prompt(client)
    test_multi_turn(client)
    test_model_echo(client)
    test_usage_fields(client)
    test_system_fingerprint(client)
    #
    _section("Sampling Parameters")
    test_temperature(client)
    test_seed_determinism(client)
    test_top_p(client)
    test_frequency_penalty(client)
    test_presence_penalty(client)
    test_max_tokens(client)
    test_stop_sequences(client)

    _section("Streaming")
    test_streaming_basic(client)
    test_streaming_finish_reason(client)

    _section("Tool Use (Function Calling)")
    test_single_tool_use(client)
    test_tool_result_roundtrip(client)
    test_mcp_tool_live_server(client)
    test_parallel_tool_use(client)
    test_parallel_tool_roundtrip(client)
    test_tool_choice_auto(client)
    test_tool_choice_required(client)
    test_tool_choice_specific(client)
    test_streaming_tool_use(client)
    test_tool_error_handling(client)
    test_agentic_loop(client)

    _section("Response Formats")
    test_json_mode(client)
    test_json_schema(client)
    test_json_schema_vision(client)
    test_n_choices(client)
    test_logprobs(client)

    _section("Multimodal / Vision")
    test_vision_input(client)
    test_mixed_content_blocks(client)

    # _section("Error Handling")
    # test_error_invalid_model(client)
    # test_error_empty_messages(client)

    print("\n" + "=" * 60)
    total = _pass + _fail + _skip
    print(f"  Results: {_pass} passed, {_fail} failed, {_skip} skipped / {total} total")
    print("=" * 60)

    log_path = _save_log()
    print(f"\n  All API responses saved to: {log_path}")
    print(f"  (paste this file to debug failures)\n")
    return _fail == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LOCAL_BASE_URL", "http://0.0.0.0:8000/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy-key-for-local"),
    )

    success = run_all(client)
    raise SystemExit(0 if success else 1)
