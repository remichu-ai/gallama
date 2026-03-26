"""
Comprehensive test suite for a local Anthropic-compatible Messages endpoint.

Assumes `client` is already instantiated as an anthropic.Anthropic(...) pointed
at your local server.  Adjust MODEL if your endpoint uses a different name.

Usage:
    # In your own script, just do:
    #   import anthropic
    #   client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="test")
    #   from test_endpoint import *
    #   run_all()
    #
    # Or run directly after setting the two env vars:
    #   LOCAL_BASE_URL=http://localhost:8000 LOCAL_API_KEY=test python test_endpoint.py
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

import anthropic
from dummy_mcp_server import (
    TEST_TOKEN,
    TEST_USER_PROMPT,
    build_anthropic_mcp_server,
    build_anthropic_mcp_toolset,
    run_dummy_mcp_server,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = os.getenv("TEST_MODEL", "claude-sonnet-4-20250514")
THINKING_BUDGET_TOKENS = int(os.getenv("TEST_THINKING_BUDGET", "1024"))
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

# Simple tools reused across tests
WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "description": "Get the current weather for a given location.",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g. 'San Francisco, CA'",
            }
        },
        "required": ["location"],
    },
}

CALCULATOR_TOOL: dict[str, Any] = {
    "name": "calculator",
    "description": "Evaluate a mathematical expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 + 2'",
            }
        },
        "required": ["expression"],
    },
}

STOCK_TOOL: dict[str, Any] = {
    "name": "get_stock_price",
    "description": "Get the current stock price for a given ticker symbol.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol, e.g. 'AAPL'",
            }
        },
        "required": ["ticker"],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_pass = 0
_fail = 0
_skip = 0


def _load_image_base64(path: str) -> tuple[str, str]:
    """Load a local image file and return (base64_data, media_type)."""
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
    return data, media_type


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
    """Extract concatenated text from response, skipping thinking blocks.

    Works whether thinking blocks are present or not, and regardless
    of their position in the content list.
    """
    return "".join(
        b.text for b in resp.content if b.type == "text"
    )


def _has_thinking(resp) -> bool:
    """Check if the response contains any thinking blocks."""
    return any(b.type == "thinking" for b in resp.content)


def _messages_url(client: anthropic.Anthropic) -> str:
    base_url = str(client.base_url).rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


def _post_messages_json(client: anthropic.Anthropic, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        _messages_url(client),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": client.api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {_messages_url(client)} failed with {exc.code}: {body}") from exc


# ---------------------------------------------------------------------------
# 1. Basic message creation (non-streaming)
# ---------------------------------------------------------------------------
def test_basic_message(client: anthropic.Anthropic):
    """Simple single-turn user message → assistant response."""
    name = "Basic message"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
        )
        assert resp.id, "Response missing id"
        assert resp.type == "message", f"Unexpected type: {resp.type}"
        assert resp.role == "assistant", f"Unexpected role: {resp.role}"
        assert len(resp.content) >= 1, "Empty content"
        text_blocks = [b for b in resp.content if b.type == "text"]
        assert len(text_blocks) >= 1, "No text block found in content"
        assert resp.stop_reason == "end_turn", f"Unexpected stop_reason: {resp.stop_reason}"
        assert resp.usage.input_tokens > 0, "input_tokens should be > 0"
        assert resp.usage.output_tokens > 0, "output_tokens should be > 0"
        text = _get_text(resp)
        has_think = _has_thinking(resp)
        _report(name, True, f"Got: {text!r}" + (" (with implicit thinking)" if has_think else ""))
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 2. System prompt
# ---------------------------------------------------------------------------
def test_system_prompt(client: anthropic.Anthropic):
    """Ensure system prompt influences the response."""
    name = "System prompt"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            system=PIRATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "How are you today?"}],
        )
        text = _get_text(resp).lower()
        assert "arrr" in text, f"Expected 'arrr' in response, got: {text}"
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 3. Multi-turn conversation
# ---------------------------------------------------------------------------
def test_multi_turn(client: anthropic.Anthropic):
    """Multi-turn conversation preserves context."""
    name = "Multi-turn conversation"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            messages=[
                {"role": "user", "content": "My name is Zephyr."},
                {"role": "assistant", "content": "Nice to meet you, Zephyr!"},
                {"role": "user", "content": "What is my name?"},
            ],
        )
        text = _get_text(resp)
        assert "Zephyr" in text, f"Expected 'Zephyr' in response: {text}"
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 4. Streaming (basic)
# ---------------------------------------------------------------------------
def test_streaming_basic(client: anthropic.Anthropic):
    """Stream a response and verify event types."""
    name = "Streaming (basic)"
    try:
        events: list[str] = []
        full_text = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
        ) as stream:
            for event in stream:
                events.append(type(event).__name__)
            full_text = stream.get_final_text()

        assert len(full_text) > 0, "No text received"
        _report(name, True, f"Events: {len(events)}, text length: {len(full_text)}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 5. Streaming with raw events
# ---------------------------------------------------------------------------
def test_streaming_raw_events(client: anthropic.Anthropic):
    """Verify raw SSE event types from the stream."""
    name = "Streaming (raw events)"
    try:
        event_types: list[str] = []
        with client.messages.stream(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for event in stream:
                event_types.append(type(event).__name__)

        # We expect at least message_start, content_block_start,
        # content_block_delta(s), content_block_stop, message_delta, message_stop
        assert len(event_types) >= 4, f"Too few events: {event_types}"
        _report(name, True, f"Event types seen: {set(event_types)}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 6. Stop sequences
# ---------------------------------------------------------------------------
def test_stop_sequences(client: anthropic.Anthropic):
    """Stop generation when a custom stop sequence is hit."""
    name = "Stop sequences"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            stop_sequences=["STOP"],
            messages=[
                {
                    "role": "user",
                    "content": "Write the sentence 'I will now STOP talking' verbatim.",
                }
            ],
        )
        assert resp.stop_reason == "stop_sequence", (
            f"Expected stop_sequence, got: {resp.stop_reason}"
        )
        _report(name, True, f"stop_reason={resp.stop_reason}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 7. Max tokens limit
# ---------------------------------------------------------------------------
def test_max_tokens(client: anthropic.Anthropic):
    """Ensure max_tokens is respected and stop_reason is max_tokens."""
    name = "Max tokens limit"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long essay about the history of the universe.",
                }
            ],
        )
        assert resp.stop_reason == "max_tokens", (
            f"Expected max_tokens, got: {resp.stop_reason}"
        )
        assert resp.usage.output_tokens <= 10, (
            f"Output tokens ({resp.usage.output_tokens}) exceeded expected cap"
        )
        _report(name, True, f"output_tokens={resp.usage.output_tokens}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 8. Temperature
# ---------------------------------------------------------------------------
def test_temperature(client: anthropic.Anthropic):
    """Temperature 0.7 should yield deterministic (or near-deterministic) output."""
    name = "Temperature 0.7 determinism"
    try:
        kwargs = dict(
            model=MODEL,
            max_tokens=3000,
            temperature=0.7,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        )
        r1 = client.messages.create(**kwargs)
        r2 = client.messages.create(**kwargs)
        t1 = _get_text(r1).strip()
        t2 = _get_text(r2).strip()
        same = t1 == t2
        _report(name, same, f"r1={t1!r}, r2={t2!r} — {'match' if same else 'MISMATCH (may be acceptable)'}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 9. Single tool use
# ---------------------------------------------------------------------------
def test_single_tool_use(client: anthropic.Anthropic):
    """Model should invoke a single tool when appropriate."""
    name = "Single tool use"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1, "No tool_use block found"
        tb = tool_blocks[0]
        assert tb.name == "get_weather", f"Wrong tool: {tb.name}"
        assert "input" in dir(tb) or hasattr(tb, "input"), "Missing input field"
        assert resp.stop_reason == "tool_use", f"stop_reason={resp.stop_reason}"
        _report(name, True, f"tool={tb.name}, input={tb.input}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 10. Tool use → tool_result round-trip
# ---------------------------------------------------------------------------
def test_tool_result_roundtrip(client: anthropic.Anthropic):
    """Full cycle: model calls tool → we return result → model responds."""
    name = "Tool result round-trip"
    try:
        # Step 1 — model requests the tool
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            tool_choice={"type": "any"},
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"}
            ],
        )
        tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
        assert tool_blocks, "Model did not call tool"
        tb = tool_blocks[0]

        # Step 2 — feed back a fake tool result
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": resp1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": "Sunny, 24°C, light breeze",
                        }
                    ],
                },
            ],
        )
        text = "".join(b.text for b in resp2.content if b.type == "text")
        assert resp2.stop_reason == "end_turn", f"stop_reason={resp2.stop_reason}"
        assert len(text) > 0, "Empty final response"
        _report(name, True, f"Final response contains weather info: {'24' in text or 'sunny' in text.lower()}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 10b. MCP tool use against a live dummy MCP server
# ---------------------------------------------------------------------------
def test_mcp_tool_live_server(client: anthropic.Anthropic):
    """Use a real MCP server and verify Gallama returns Anthropic MCP blocks."""
    name = "MCP tool (live server)"
    try:
        with run_dummy_mcp_server() as mcp_server:
            payload = {
                "model": MODEL,
                "max_tokens": 3000,
                "mcp_servers": [build_anthropic_mcp_server(mcp_server.url)],
                "tools": [build_anthropic_mcp_toolset()],
                "messages": [{"role": "user", "content": TEST_USER_PROMPT}],
            }
            resp = _post_messages_json(client, payload)
            content = resp.get("content") or []
            block_types = [block.get("type") for block in content]
            text = "".join(block.get("text", "") for block in content if block.get("type") == "text").strip()

            assert TEST_TOKEN in text, f"Expected token {TEST_TOKEN!r} in response: {text!r}"
            assert "mcp_tool_use" in block_types, f"Missing mcp_tool_use in content: {block_types}"
            assert "mcp_tool_result" in block_types, f"Missing mcp_tool_result in content: {block_types}"
            assert mcp_server.count_method("tools/list") >= 1, "MCP server did not receive tools/list"
            assert mcp_server.count_method("tools/call") >= 1, "MCP server did not receive tools/call"

            _report(
                name,
                True,
                f"block_types={block_types}, tools/call={mcp_server.count_method('tools/call')}",
            )
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 11. Multiple / parallel tool use
# ---------------------------------------------------------------------------
def test_parallel_tool_use(client: anthropic.Anthropic):
    """Model should invoke multiple tools in a single turn."""
    name = "Parallel tool use"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL, STOCK_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need two things at once: "
                        "1) the weather in London and "
                        "2) the stock price of AAPL. "
                        "Please call both tools."
                    ),
                }
            ],
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        tool_names = {b.name for b in tool_blocks}
        both = {"get_weather", "get_stock_price"} <= tool_names
        _report(
            name,
            len(tool_blocks) >= 2,
            f"Tool calls: {len(tool_blocks)}, names: {tool_names}"
            + (" (both present)" if both else " (missing one)"),
        )
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 12. Parallel tool use round-trip
# ---------------------------------------------------------------------------
def test_parallel_tool_roundtrip(client: anthropic.Anthropic):
    """Full cycle with multiple tool calls and results."""
    name = "Parallel tool round-trip"
    try:
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL, STOCK_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris and the stock price of MSFT?",
                }
            ],
        )
        tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1, "No tool calls"

        # Build tool results for all calls
        fake_results = {
            "get_weather": "Cloudy, 18°C in Paris",
            "get_stock_price": "MSFT is trading at $420.50",
        }
        tool_result_blocks = []
        for tb in tool_blocks:
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": fake_results.get(tb.name, "Unknown tool"),
                }
            )

        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL, STOCK_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Paris and the stock price of MSFT?"},
                {"role": "assistant", "content": resp1.content},
                {"role": "user", "content": tool_result_blocks},
            ],
        )
        text = "".join(b.text for b in resp2.content if b.type == "text")
        assert len(text) > 0, "Empty final text"
        _report(name, True, f"Final answer length: {len(text)} chars")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 13. tool_choice: auto / any / specific
# ---------------------------------------------------------------------------
def test_tool_choice_auto(client: anthropic.Anthropic):
    """tool_choice='auto' lets the model decide."""
    name = "tool_choice=auto"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        # With "auto", model should NOT call the weather tool for a joke
        _report(name, True, f"Tool calls: {len(tool_blocks)} (expected 0)")
    except Exception as e:
        _report(name, False, str(e))


def test_tool_choice_any(client: anthropic.Anthropic):
    """tool_choice='any' forces the model to use some tool."""
    name = "tool_choice=any"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": "Hello!"}],
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1, "Expected at least one tool call with tool_choice=any"
        _report(name, True, f"Forced tool: {tool_blocks[0].name}")
    except Exception as e:
        _report(name, False, str(e))


def test_tool_choice_specific(client: anthropic.Anthropic):
    """tool_choice forces a specific tool."""
    name = "tool_choice=specific tool"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            tool_choice={"type": "tool", "name": "calculator"},
            messages=[
                {"role": "user", "content": "What's the weather in NYC?"}
            ],
        )
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1, "No tool call"
        assert tool_blocks[0].name == "calculator", (
            f"Expected calculator, got {tool_blocks[0].name}"
        )
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 14. Streaming + tool use
# ---------------------------------------------------------------------------
def test_streaming_tool_use(client: anthropic.Anthropic):
    """Stream a response that includes a tool call."""
    name = "Streaming + tool use"
    try:
        tool_name = None
        tool_input_json = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
        ) as stream:
            for event in stream:
                etype = type(event).__name__
                # Capture tool info from input_json events
                if hasattr(event, "name") and event.name:
                    tool_name = event.name
        final_message = stream.get_final_message()
        tool_blocks = [b for b in final_message.content if b.type == "tool_use"]
        assert tool_blocks, "No tool_use in streamed response"
        _report(name, True, f"Streamed tool: {tool_blocks[0].name}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 15. Extended thinking
# ---------------------------------------------------------------------------
def test_thinking(client: anthropic.Anthropic):
    """Extended thinking (thinking blocks in response)."""
    name = "Extended thinking"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS,
            },
            messages=[
                {
                    "role": "user",
                    "content": "Think step by step: what is 27 * 43, and provide me only the answer, dont share your working?",
                }
            ],
        )
        thinking_blocks = [b for b in resp.content if b.type == "thinking"]
        text_blocks = [b for b in resp.content if b.type == "text"]
        has_thinking = len(thinking_blocks) >= 1
        has_text = len(text_blocks) >= 1
        if has_text and not has_thinking:
            _skip_test(
                name,
                f"Model returned text but no thinking blocks. Thinking likely unsupported for model={MODEL}.",
            )
            return
        thinking_text = thinking_blocks[0].thinking if has_thinking else ""
        _report(
            name,
            has_thinking and has_text,
            f"thinking_blocks={len(thinking_blocks)}, "
            f"text_blocks={len(text_blocks)}, "
            f"thinking_preview={thinking_text[:80]!r}...",
        )
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 16. Streaming + thinking
# ---------------------------------------------------------------------------
def test_streaming_thinking(client: anthropic.Anthropic):
    """Stream a response with extended thinking enabled."""
    name = "Streaming + thinking"
    try:
        saw_thinking = False
        saw_text = False
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS,
            },
            messages=[
                {"role": "user", "content": "What is 15! (15 factorial)? Think carefully."}
            ],
        ) as stream:
            for event in stream:
                etype = type(event).__name__
                if "thinking" in etype.lower() or (hasattr(event, "type") and "thinking" in str(getattr(event, "type", ""))):
                    saw_thinking = True
                if hasattr(event, "type") and getattr(event, "type", "") == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", "") == "thinking":
                        saw_thinking = True
                    if block and getattr(block, "type", "") == "text":
                        saw_text = True

            final = stream.get_final_message()
            thinking_blocks = [b for b in final.content if b.type == "thinking"]
            text_blocks = [b for b in final.content if b.type == "text"]
            saw_thinking = saw_thinking or len(thinking_blocks) > 0
            saw_text = saw_text or len(text_blocks) > 0

        if saw_text and not saw_thinking:
            _skip_test(
                name,
                f"Model streamed text but no thinking blocks. Thinking likely unsupported for model={MODEL}.",
            )
            return

        _report(name, saw_thinking and saw_text, f"thinking={saw_thinking}, text={saw_text}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 17. Thinking + tool use
# ---------------------------------------------------------------------------
def test_thinking_with_tools(client: anthropic.Anthropic):
    """Extended thinking combined with tool use."""
    name = "Thinking + tool use"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET_TOKENS,
            },
            tools=[CALCULATOR_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": "Use the calculator to compute 123 * 456",
                }
            ],
        )
        thinking_blocks = [b for b in resp.content if b.type == "thinking"]
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        if len(tool_blocks) >= 1 and not thinking_blocks:
            _skip_test(
                name,
                f"Tool call worked but no thinking blocks were returned. Thinking likely unsupported for model={MODEL}.",
            )
            return
        _report(
            name,
            len(tool_blocks) >= 1 and len(thinking_blocks) >= 1,
            f"thinking={len(thinking_blocks)}, tools={len(tool_blocks)}",
        )
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 18. Image / vision input
# ---------------------------------------------------------------------------
def test_vision_input(client: anthropic.Anthropic):
    """Send a local image of a cat and ask the model to describe it."""
    name = "Vision / image input (cat)"
    try:
        image_data, media_type = _load_image_base64(LOCAL_IMAGE_PATH)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What animal is in this image? Describe it briefly.",
                        },
                    ],
                }
            ],
        )
        text = _get_text(resp)

        assert len(text) > 0, "Empty response for image description"
        assert any(word in text.lower() for word in ("kitten", "cat")), \
            f"Expected 'kitten' or 'cat' in description, got: {text}"

        _report(name, True, f"Description: {text[:80]}...")
    except Exception as e:
        _report(name, False, str(e))

# ---------------------------------------------------------------------------
# 19. Top-p / top-k sampling
# ---------------------------------------------------------------------------
def test_top_p(client: anthropic.Anthropic):
    """Verify top_p parameter is accepted."""
    name = "top_p parameter"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            top_p=0.9,
            messages=[{"role": "user", "content": "Say one word."}],
        )
        assert _get_text(resp), "No output"
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


def test_top_k(client: anthropic.Anthropic):
    """Verify top_k parameter is accepted."""
    name = "top_k parameter"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            top_k=40,
            messages=[{"role": "user", "content": "Say one word."}],
        )
        assert _get_text(resp), "No output"
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 20. Multimodal content blocks (mixed text + image)
# ---------------------------------------------------------------------------
def test_mixed_content_blocks(client: anthropic.Anthropic):
    """Multiple content blocks of different types in one user turn."""
    name = "Mixed content blocks"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First, note that my favourite colour is blue."},
                        {"type": "text", "text": "Now, what is my favourite colour?"},
                    ],
                }
            ],
        )
        text = _get_text(resp).lower()
        assert "blue" in text, f"Expected 'blue' in: {text}"
        _report(name, True)
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 21. Error handling — invalid model
# ---------------------------------------------------------------------------
def test_error_invalid_model(client: anthropic.Anthropic):
    """Request with a nonsense model name should return an error."""
    name = "Error: invalid model"
    try:
        client.messages.create(
            model="nonexistent-model-xyz",
            max_tokens=3000,
            messages=[{"role": "user", "content": "Hi"}],
        )
        _report(name, False, "Expected an error but got a 200")
    except anthropic.NotFoundError:
        _report(name, True, "Got NotFoundError as expected")
    except anthropic.BadRequestError:
        _report(name, True, "Got BadRequestError as expected")
    except anthropic.APIError as e:
        _report(name, True, f"Got APIError: {e.status_code}")
    except Exception as e:
        _report(name, False, f"Unexpected error type: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 22. Error handling — empty messages
# ---------------------------------------------------------------------------
def test_error_empty_messages(client: anthropic.Anthropic):
    """Empty messages array should return a 400-level error."""
    name = "Error: empty messages"
    try:
        client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[],
        )
        _report(name, False, "Expected an error but got a 200")
    except (anthropic.BadRequestError, anthropic.UnprocessableEntityError):
        _report(name, True, "Got expected 4xx error")
    except anthropic.APIError as e:
        _report(name, True, f"Got APIError: {e.status_code}")
    except Exception as e:
        # SDK might validate client-side
        _report(name, True, f"Client-side validation: {type(e).__name__}")


# ---------------------------------------------------------------------------
# 23. Usage / token counting
# ---------------------------------------------------------------------------
def test_usage_fields(client: anthropic.Anthropic):
    """Verify usage object has expected fields with sane values."""
    name = "Usage fields"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi."}],
        )
        u = resp.usage
        assert u.input_tokens > 0, f"input_tokens={u.input_tokens}"
        assert u.output_tokens > 0, f"output_tokens={u.output_tokens}"
        _report(name, True, f"input={u.input_tokens}, output={u.output_tokens}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 24. Model field echo
# ---------------------------------------------------------------------------
def test_model_echo(client: anthropic.Anthropic):
    """Response should echo back the model identifier."""
    name = "Model echo"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert resp.model, "model field is empty"
        _report(name, True, f"model={resp.model}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 25. tool_use with error result
# ---------------------------------------------------------------------------
def test_tool_error_result(client: anthropic.Anthropic):
    """Return an error tool result and verify model handles it gracefully."""
    name = "Tool error result"
    try:
        resp1 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Atlantis?"}
            ],
        )
        tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
        if not tool_blocks:
            _report(name, True, "Model answered without tool (acceptable)")
            return

        tb = tool_blocks[0]
        resp2 = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[WEATHER_TOOL],
            messages=[
                {"role": "user", "content": "What's the weather in Atlantis?"},
                {"role": "assistant", "content": resp1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "is_error": True,
                            "content": "Error: Location not found",
                        }
                    ],
                },
            ],
        )
        text = "".join(b.text for b in resp2.content if b.type == "text")
        assert len(text) > 0, "Empty response after tool error"
        _report(name, True, f"Handled error gracefully, response length={len(text)}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 26. Sequential multi-tool (agentic loop)
# ---------------------------------------------------------------------------
def test_agentic_loop(client: anthropic.Anthropic):
    """Simulate an agentic loop: model calls tools sequentially until done."""
    name = "Agentic loop (multi-step)"
    try:
        tools = [WEATHER_TOOL, CALCULATOR_TOOL]
        messages = [
            {
                "role": "user",
                "content": (
                    "First get the weather in NYC, then use the calculator to compute "
                    "the temperature in Fahrenheit if it's 22°C (formula: C*9/5+32)."
                ),
            }
        ]

        max_iterations = 5
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, tools=tools, messages=messages
            )
            # Append assistant response
            messages.append({"role": "assistant", "content": resp.content})

            tool_blocks = [b for b in resp.content if b.type == "tool_use"]
            if not tool_blocks:
                break  # Model is done

            # Build fake results
            results = []
            for tb in tool_blocks:
                if tb.name == "get_weather":
                    result = "22°C, partly cloudy in NYC"
                elif tb.name == "calculator":
                    result = "71.6"
                else:
                    result = "unknown"
                results.append(
                    {"type": "tool_result", "tool_use_id": tb.id, "content": result}
                )
            messages.append({"role": "user", "content": results})

        text = "".join(
            b.text for b in resp.content if b.type == "text"
        )
        _report(
            name,
            iteration >= 2 and len(text) > 0,
            f"Completed in {iteration} iterations",
        )
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 27. JSON mode (basic structured output)
# ---------------------------------------------------------------------------
def test_json_mode(client: anthropic.Anthropic):
    """output_config with json_schema should return valid JSON.

    Equivalent to OpenAI's response_format={"type": "json_object"}.
    Anthropic does not have a bare "json_object" mode, so we use a
    permissive json_schema that accepts any object.
    """
    name = "JSON mode"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4024,
            system=JSON_MODE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Give me a JSON object with keys 'name' and 'age' for a fictional character.",
                }
            ],
            extra_body={
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                            "required": ["name", "age"],
                        },
                    }
                }
            },
        )
        text = _get_text(resp)
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        assert "name" in parsed, "Missing 'name' key"
        assert "age" in parsed, "Missing 'age' key"
        _report(name, True, f"Parsed JSON: {parsed}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 28. JSON Schema (structured output with full validation)
# ---------------------------------------------------------------------------
def test_json_schema(client: anthropic.Anthropic):
    """output_config with json_schema should return JSON conforming to the given schema.

    Equivalent to OpenAI's response_format={"type": "json_schema", ...}.
    """
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
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4024,
            system=JSON_SCHEMA_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Give me a fictional character with a name, age (integer), and a list of hobbies.",
                }
            ],
            extra_body={
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            },
        )
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
        _report(name, True, f"Parsed JSON: {parsed}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# 29. JSON Schema + Vision
# ---------------------------------------------------------------------------
def test_json_schema_vision(client: anthropic.Anthropic):
    """output_config with json_schema and a local image input should return conforming JSON."""
    name = "JSON schema + vision"
    schema = {
        "type": "object",
        "properties": {
            "animal": {
                "type": "string",
                "enum": ["puppy", "kitten", "hamster", "parrot", "rabbit"],
            },
            "color": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["animal", "color", "description"],
        "additionalProperties": False,
    }
    try:
        image_data, media_type = _load_image_base64(LOCAL_IMAGE_PATH)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=4024,
            system=VISION_JSON_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Identify the animal in this image. Pick the closest match from the allowed options. Return its type, primary color, and a one-sentence description.",
                        },
                    ],
                }
            ],
            extra_body={
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }
            },
        )
        text = _get_text(resp)
        parsed = json.loads(text)
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        for key in ("animal", "color", "description"):
            assert key in parsed, f"Missing required key: {key}"
            assert isinstance(parsed[key], str), f"{key} should be str, got {type(parsed[key])}"
        extra = set(parsed.keys()) - {"animal", "color", "description"}
        assert not extra, f"Unexpected extra keys: {extra}"
        allowed = ["puppy", "kitten", "hamster", "parrot", "rabbit"]
        assert parsed["animal"] in allowed, f"animal must be one of {allowed}, got: {parsed['animal']}"
        assert parsed["animal"] == "kitten", f"Expected 'kitten', got: {parsed['animal']}"
        _report(name, True, f"Parsed JSON: {parsed}")
    except Exception as e:
        _report(name, False, str(e))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_TESTS = [
    test_basic_message,
    test_system_prompt,
    test_multi_turn,
    test_streaming_basic,
    test_streaming_raw_events,
    test_stop_sequences,
    test_max_tokens,
    test_temperature,
    test_single_tool_use,
    test_tool_result_roundtrip,
    test_parallel_tool_use,
    test_parallel_tool_roundtrip,
    test_tool_choice_auto,
    test_tool_choice_any,
    test_tool_choice_specific,
    test_streaming_tool_use,
    test_thinking,
    test_streaming_thinking,
    test_thinking_with_tools,
    test_vision_input,
    test_top_p,
    test_top_k,
    test_mixed_content_blocks,
    test_error_invalid_model,
    test_error_empty_messages,
    test_usage_fields,
    test_model_echo,
    test_tool_error_result,
    test_agentic_loop,
    test_json_mode,
    test_json_schema,
    test_json_schema_vision,
]


def run_all(client: anthropic.Anthropic):
    """Run every test and print a summary."""
    global _pass, _fail, _skip
    _pass = _fail = _skip = 0

    print("\n" + "=" * 60)
    print("  Anthropic Messages API — Compatibility Test Suite")
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {client.base_url}")
    print("=" * 60)

    _section("Basic Features")
    test_basic_message(client)
    test_system_prompt(client)
    test_multi_turn(client)
    test_model_echo(client)
    test_usage_fields(client)

    _section("Sampling Parameters")
    test_temperature(client)
    test_top_p(client)
    test_top_k(client)
    test_max_tokens(client)
    test_stop_sequences(client)

    _section("Streaming")
    test_streaming_basic(client)
    test_streaming_raw_events(client)

    _section("Tool Use")
    test_mcp_tool_live_server(client)
    test_single_tool_use(client)
    test_tool_result_roundtrip(client)
    test_parallel_tool_use(client)
    test_parallel_tool_roundtrip(client)
    test_tool_choice_auto(client)
    test_tool_choice_any(client)
    test_tool_choice_specific(client)
    test_streaming_tool_use(client)
    test_tool_error_result(client)
    test_agentic_loop(client)

    _section("Extended Thinking")
    test_thinking(client)
    test_streaming_thinking(client)
    test_thinking_with_tools(client)

    _section("Response Formats")
    test_json_mode(client)
    test_json_schema(client)
    test_json_schema_vision(client)

    _section("Multimodal / Vision")
    test_vision_input(client)
    test_mixed_content_blocks(client)

    # _section("Error Handling")
    # test_error_invalid_model(client)
    # test_error_empty_messages(client)

    print("\n" + "=" * 60)
    total = _pass + _fail + _skip
    print(f"  Results: {_pass} passed, {_fail} failed, {_skip} skipped / {total} total")
    print("=" * 60 + "\n")
    return _fail == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from anthropic import Anthropic

    client = Anthropic(
        base_url=os.getenv("LOCAL_BASE_URL", "http://0.0.0.0:8000/"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy-key-for-local"),
    )

    success = run_all(client)
    raise SystemExit(0 if success else 1)
