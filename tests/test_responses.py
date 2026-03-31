"""
Smoke test suite for a local OpenAI-compatible Responses API endpoint.

Usage:
    LOCAL_BASE_URL=http://localhost:8000/v1 LOCAL_API_KEY=test python src/tests/test_responses.py

Optional environment variables:
    TEST_MODEL=my-model
    TEST_LOG_FILE=test_responses_api.json
    TEST_IMAGE_PATH=cat1.jpg
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

import openai
from dummy_mcp_server import (
    TEST_TOKEN,
    TEST_USER_PROMPT,
    build_responses_mcp_tool,
    run_dummy_mcp_server,
)


MODEL = os.getenv("TEST_MODEL", "gpt-4o")
LOG_FILE = os.getenv("TEST_LOG_FILE", "test_responses_api.json")
COMMON_INSTRUCTIONS = os.getenv(
    "TEST_SYSTEM_PROMPT",
    "You are a helpful assistant. Follow the user's instructions exactly.",
)
PIRATE_INSTRUCTIONS = (
    f"{COMMON_INSTRUCTIONS}\n\nYou are a pirate. Every answer must contain 'Arrr'."
)
JSON_SCHEMA_INSTRUCTIONS = (
    f"{COMMON_INSTRUCTIONS}\n\nReturn only JSON that matches the requested schema."
)
LOCAL_IMAGE_PATH = os.getenv("TEST_IMAGE_PATH", "cat1.jpg")

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
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
}

STOCK_TOOL: dict[str, Any] = {
    "type": "function",
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
}

_log_entries: list[dict[str, Any]] = []
_pass = 0
_fail = 0
_skip = 0


def _load_image_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(path, "rb") as handle:
        data = base64.standard_b64encode(handle.read()).decode("utf-8")
    return f"data:{media_type};base64,{data}"


def _resp_to_dict(resp) -> dict[str, Any] | None:
    if resp is None:
        return None
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
    stream_events: list[Any] | None = None,
    error: Exception | None = None,
    passed: bool,
    detail: str = "",
):
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
    if stream_events is not None:
        entry["stream_events"] = [_resp_to_dict(event) for event in stream_events]
    if error is not None:
        entry["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        }
        if hasattr(error, "response") and error.response is not None:
            try:
                entry["error"]["http_status"] = error.response.status_code
                entry["error"]["response_body"] = error.response.text
            except Exception:
                pass
    _log_entries.append(entry)


def _save_log():
    path = os.path.abspath(LOG_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": MODEL,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "total_tests": len(_log_entries),
                "passed": sum(1 for entry in _log_entries if entry["passed"]),
                "failed": sum(1 for entry in _log_entries if not entry["passed"]),
                "entries": _log_entries,
            },
            handle,
            indent=2,
            default=str,
        )
    return path


def _report(name: str, passed: bool, detail: str = ""):
    global _pass, _fail
    prefix = "[PASS]" if passed else "[FAIL]"
    if passed:
        _pass += 1
    else:
        _fail += 1
    print(f"  {prefix} {name}" + (f" - {detail}" if detail else ""))


def _skip_test(name: str, reason: str):
    global _skip
    _skip += 1
    print(f"  [SKIP] {name} - {reason}")


def _section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_output_items(resp) -> list[Any]:
    output = _get_value(resp, "output")
    return list(output or [])


def _get_output_text(resp) -> str:
    output_text = _get_value(resp, "output_text")
    if output_text:
        return output_text

    text_parts: list[str] = []
    for item in _get_output_items(resp):
        if _get_value(item, "type") != "message":
            continue
        for part in _get_value(item, "content", []) or []:
            if _get_value(part, "type") == "output_text":
                text_parts.append(_get_value(part, "text", "") or "")
    return "".join(text_parts)


def _get_function_calls(resp) -> list[Any]:
    return [item for item in _get_output_items(resp) if _get_value(item, "type") == "function_call"]


def _get_reasoning_text(resp) -> str:
    reasoning_parts: list[str] = []
    for item in _get_output_items(resp):
        if _get_value(item, "type") != "reasoning":
            continue
        for part in _get_value(item, "content", []) or []:
            if _get_value(part, "type") == "reasoning_text":
                text = _get_value(part, "text", "") or ""
                if text:
                    reasoning_parts.append(text)
        for part in _get_value(item, "summary", []) or []:
            if _get_value(part, "type") == "summary_text":
                text = _get_value(part, "text", "") or ""
                if text:
                    reasoning_parts.append(text)
    return "\n".join(reasoning_parts)


def _request_json(client: openai.OpenAI, method: str, path: str) -> dict[str, Any]:
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}{path}"
    headers = {"Authorization": f"Bearer {client.api_key}"}
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {body}") from exc


def _retrieve_response(client: openai.OpenAI, response_id: str) -> Any:
    responses_api = client.responses
    if hasattr(responses_api, "retrieve"):
        return responses_api.retrieve(response_id)
    return _request_json(client, "GET", f"/responses/{response_id}")


def _delete_response(client: openai.OpenAI, response_id: str) -> Any:
    responses_api = client.responses
    if hasattr(responses_api, "delete"):
        result = responses_api.delete(response_id)
        if result is None:
            return {
                "id": response_id,
                "object": "response.deleted",
                "deleted": True,
            }
        return result
    return _request_json(client, "DELETE", f"/responses/{response_id}")


def test_basic_response(client: openai.OpenAI):
    name = "Basic response"
    params = {
        "model": MODEL,
        "input": "Say hello in exactly 3 words.",
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        text = _get_output_text(resp)
        assert resp.id, "Response missing id"
        assert resp.object == "response", f"Unexpected object: {resp.object}"
        assert resp.status == "completed", f"Unexpected status: {resp.status}"
        assert text, "Empty output_text"
        assert resp.usage.input_tokens > 0, "input_tokens should be > 0"
        assert resp.usage.output_tokens > 0, "output_tokens should be > 0"
        _report(name, True, f"Got: {text!r}")
        _log_response(name, request_params=params, response=resp, passed=True, detail=f"Got: {text!r}")
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_instructions(client: openai.OpenAI):
    name = "Instructions"
    params = {
        "model": MODEL,
        "instructions": PIRATE_INSTRUCTIONS,
        "input": "How are you today?",
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        text = _get_output_text(resp).lower()
        assert "arrr" in text, f"Expected 'arrr' in response, got: {text}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_multi_turn(client: openai.OpenAI):
    name = "Multi-turn conversation"
    params = {
        "model": MODEL,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "My name is Zephyr."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Nice to meet you, Zephyr!"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is my name?"}]},
        ],
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        text = _get_output_text(resp)
        assert "Zephyr" in text, f"Expected 'Zephyr' in response: {text}"
        _report(name, True)
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_reasoning_output(client: openai.OpenAI):
    name = "Reasoning output"
    params = {
        "model": MODEL,
        "input": "What is 27 multiplied by 14? Answer with just the number.",
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        reasoning_text = _get_reasoning_text(resp)
        assert reasoning_text, "No reasoning output items found"
        detail = f"Reasoning length: {len(reasoning_text)}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_streaming_basic(client: openai.OpenAI):
    name = "Streaming (basic)"
    params = {
        "model": MODEL,
        "input": "Count from 1 to 5.",
        "max_output_tokens": 300,
        "stream": True,
    }
    events = []
    try:
        full_text = ""
        stream = client.responses.create(**params)
        for event in stream:
            events.append(event)
            if getattr(event, "type", None) == "response.output_text.delta":
                full_text += event.delta or ""

        assert full_text, "No streamed text received"
        _report(name, True, f"Events: {len(events)}, text length: {len(full_text)}")
        _log_response(name, request_params=params, stream_events=events, passed=True, detail=f"Events: {len(events)}")
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, stream_events=events if events else None, error=exc, passed=False, detail=str(exc))


def test_previous_response_id(client: openai.OpenAI):
    name = "Stateful previous_response_id"
    secret = "cobalt-sparrow-731"
    resp1 = resp2 = retrieved = None
    try:
        params1 = {
            "model": MODEL,
            "input": f"Remember this token for later: {secret}. Reply with exactly STORED.",
            "max_output_tokens": 100,
            "store": True,
        }
        resp1 = client.responses.create(**params1)
        text1 = _get_output_text(resp1).strip().lower()
        assert "stored" in text1, f"Unexpected first response: {text1}"

        params2 = {
            "model": MODEL,
            "previous_response_id": resp1.id,
            "input": "What token did I ask you to remember? Reply with only the token.",
            "max_output_tokens": 100,
        }
        resp2 = client.responses.create(**params2)
        text2 = _get_output_text(resp2).strip().lower()
        assert secret in text2, f"Expected remembered token in: {text2}"

        retrieved = _retrieve_response(client, resp2.id)
        assert _get_value(retrieved, "id") == resp2.id, "Retrieved response id mismatch"
        assert _get_value(retrieved, "previous_response_id") == resp1.id, "previous_response_id was not persisted"

        detail = f"recalled token={text2!r}"
        _report(name, True, detail)
        _log_response(
            name,
            request_params={"step1": params1, "step2": params2},
            response={
                "step1": _resp_to_dict(resp1),
                "step2": _resp_to_dict(resp2),
                "retrieved": _resp_to_dict(retrieved),
            },
            passed=True,
            detail=detail,
        )
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(
            name,
            response={
                "step1": _resp_to_dict(resp1),
                "step2": _resp_to_dict(resp2),
                "retrieved": _resp_to_dict(retrieved),
            },
            error=exc,
            passed=False,
            detail=str(exc),
        )


def test_single_tool_use(client: openai.OpenAI):
    name = "Single tool use"
    params = {
        "model": MODEL,
        "input": "What's the weather in San Francisco?",
        "tools": [WEATHER_TOOL],
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        function_calls = _get_function_calls(resp)
        assert function_calls, "No function_call output items found"
        assert resp.status == "completed", f"Unexpected status: {resp.status}"
        assert all(_get_value(call, "status") == "completed" for call in function_calls), (
            f"Unexpected function_call status values: {[_get_value(call, 'status') for call in function_calls]}"
        )
        call = function_calls[0]
        assert call.name == "get_weather", f"Wrong tool: {call.name}"
        args = json.loads(call.arguments)
        detail = f"tool={call.name}, args={args}, response_status={resp.status}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_tool_result_roundtrip(client: openai.OpenAI):
    name = "Tool result round-trip"
    resp1 = resp2 = None
    try:
        params1 = {
            "model": MODEL,
            "input": "What's the weather in Tokyo?",
            "tools": [WEATHER_TOOL],
            "max_output_tokens": 300,
        }
        resp1 = client.responses.create(**params1)
        function_calls = _get_function_calls(resp1)
        assert function_calls, "Model did not call a function"
        assert resp1.status == "completed", f"Unexpected status: {resp1.status}"
        assert all(_get_value(item, "status") == "completed" for item in function_calls), (
            f"Unexpected function_call status values: {[_get_value(item, 'status') for item in function_calls]}"
        )
        call = function_calls[0]

        params2 = {
            "model": MODEL,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What's the weather in Tokyo?"}]},
                *_resp_to_dict(resp1)["output"],
                {"type": "function_call_output", "call_id": call.call_id, "output": "Sunny, 24C, light breeze"},
            ],
            "tools": [WEATHER_TOOL],
            "max_output_tokens": 300,
        }
        resp2 = client.responses.create(**params2)
        assert resp2.status == "completed", f"Unexpected status: {resp2.status}"
        assert not _get_function_calls(resp2), "Final response should not contain unresolved function_call items"
        text = _get_output_text(resp2)
        assert text, "Empty final response"
        _report(name, True, f"Final response length: {len(text)}")
        _log_response(
            name,
            request_params={"step1": params1, "step2": params2},
            response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)},
            passed=True,
        )
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(
            name,
            response={"step1": _resp_to_dict(resp1), "step2": _resp_to_dict(resp2)},
            error=exc,
            passed=False,
            detail=str(exc),
        )


def test_mcp_tool_live_server(client: openai.OpenAI):
    name = "MCP tool (live server)"
    resp = None
    params = None
    mcp_requests: list[dict[str, Any]] = []
    try:
        with run_dummy_mcp_server() as mcp_server:
            params = {
                "model": MODEL,
                "input": TEST_USER_PROMPT,
                "tools": [build_responses_mcp_tool(mcp_server.url)],
                "max_output_tokens": 300,
            }
            resp = client.responses.create(**params)
            mcp_requests = mcp_server.requests

            output_types = [_get_value(item, "type") for item in _get_output_items(resp)]
            text = _get_output_text(resp).strip()

            assert TEST_TOKEN in text, f"Expected token {TEST_TOKEN!r} in response: {text!r}"
            assert "mcp_list_tools" in output_types, f"Missing mcp_list_tools in output: {output_types}"
            assert "mcp_call" in output_types, f"Missing mcp_call in output: {output_types}"
            assert mcp_server.count_method("tools/list") >= 1, "MCP server did not receive tools/list"
            assert mcp_server.count_method("tools/call") >= 1, "MCP server did not receive tools/call"

            detail = (
                f"token returned, output_types={output_types}, "
                f"tools/call={mcp_server.count_method('tools/call')}"
            )
            _report(name, True, detail)
            _log_response(
                name,
                request_params=params,
                response={"response": _resp_to_dict(resp), "mcp_requests": mcp_requests},
                passed=True,
                detail=detail,
            )
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(
            name,
            request_params=params,
            response={"response": _resp_to_dict(resp), "mcp_requests": mcp_requests},
            error=exc,
            passed=False,
            detail=str(exc),
        )


def test_parallel_tool_use(client: openai.OpenAI):
    name = "Parallel tool use"
    params = {
        "model": MODEL,
        "input": "Provide both the weather in London and the stock price of AAPL. Call both tools.",
        "tools": [WEATHER_TOOL, STOCK_TOOL],
        "max_output_tokens": 400,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        function_calls = _get_function_calls(resp)
        tool_names = {call.name for call in function_calls}
        assert resp.status == "completed", f"Unexpected status: {resp.status}"
        assert all(_get_value(call, "status") == "completed" for call in function_calls), (
            f"Unexpected function_call status values: {[_get_value(call, 'status') for call in function_calls]}"
        )
        passed = {"get_weather", "get_stock_price"} <= tool_names
        detail = f"Function calls: {len(function_calls)}, names: {sorted(tool_names)}, response_status={resp.status}"
        _report(name, passed, detail)
        _log_response(name, request_params=params, response=resp, passed=passed, detail=detail)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_json_schema(client: openai.OpenAI):
    name = "JSON schema"
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "hobbies": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age", "hobbies"],
        "additionalProperties": False,
    }
    params = {
        "model": MODEL,
        "instructions": JSON_SCHEMA_INSTRUCTIONS,
        "input": "Give me a fictional character with a name, age, and a list of hobbies.",
        "max_output_tokens": 400,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fictional_character",
                "strict": True,
                "schema": schema,
            }
        },
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        parsed = json.loads(_get_output_text(resp))
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"
        for key in ("name", "age", "hobbies"):
            assert key in parsed, f"Missing key: {key}"
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)
        assert isinstance(parsed["hobbies"], list)
        detail = f"Parsed JSON keys: {sorted(parsed.keys())}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_vision_input(client: openai.OpenAI):
    name = "Vision / image input"
    if not os.path.exists(LOCAL_IMAGE_PATH):
        _skip_test(name, f"Missing image file: {LOCAL_IMAGE_PATH}")
        return

    image_data_url = _load_image_data_url(LOCAL_IMAGE_PATH)
    params = {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_data_url},
                    {"type": "input_text", "text": "What animal is in this image? Describe it briefly."},
                ],
            }
        ],
        "max_output_tokens": 300,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        text = _get_output_text(resp).lower()
        assert text, "Empty response for image description"
        assert any(word in text for word in ("cat", "kitten")), f"Expected cat/kitten in: {text}"
        _report(name, True, f"Description length: {len(text)}")
        _log_response(name, request_params=params, response=resp, passed=True)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_usage_fields(client: openai.OpenAI):
    name = "Usage fields"
    params = {
        "model": MODEL,
        "input": "Hi.",
        "max_output_tokens": 100,
    }
    resp = None
    try:
        resp = client.responses.create(**params)
        usage = resp.usage
        assert usage.input_tokens > 0, f"input_tokens={usage.input_tokens}"
        assert usage.output_tokens > 0, f"output_tokens={usage.output_tokens}"
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens, (
            f"total_tokens mismatch: {usage.total_tokens} != {usage.input_tokens} + {usage.output_tokens}"
        )
        detail = f"input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}"
        _report(name, True, detail)
        _log_response(name, request_params=params, response=resp, passed=True, detail=detail)
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(name, request_params=params, response=resp, error=exc, passed=False, detail=str(exc))


def test_retrieve_and_delete(client: openai.OpenAI):
    name = "Retrieve and delete"
    params = {
        "model": MODEL,
        "input": "Say the word durable once.",
        "max_output_tokens": 100,
        "store": True,
    }
    created = retrieved = deleted = None
    try:
        created = client.responses.create(**params)
        retrieved = _retrieve_response(client, created.id)
        assert _get_value(retrieved, "id") == created.id, "retrieve returned wrong response id"
        assert _get_output_text(retrieved), "retrieved response is missing output text"

        deleted = _delete_response(client, created.id)
        assert _get_value(deleted, "deleted") is True, f"Unexpected delete payload: {deleted}"
        assert _get_value(deleted, "id") == created.id, "delete returned wrong response id"

        missing = False
        try:
            _retrieve_response(client, created.id)
        except Exception:
            missing = True
        assert missing, "Deleted response was still retrievable"

        _report(name, True)
        _log_response(
            name,
            request_params=params,
            response={
                "created": _resp_to_dict(created),
                "retrieved": _resp_to_dict(retrieved),
                "deleted": _resp_to_dict(deleted),
            },
            passed=True,
        )
    except Exception as exc:
        _report(name, False, str(exc))
        _log_response(
            name,
            request_params=params,
            response={
                "created": _resp_to_dict(created),
                "retrieved": _resp_to_dict(retrieved),
                "deleted": _resp_to_dict(deleted),
            },
            error=exc,
            passed=False,
            detail=str(exc),
        )


def run_all(client: openai.OpenAI):
    global _pass, _fail, _skip
    _pass = _fail = _skip = 0
    _log_entries.clear()

    print("\n" + "=" * 60)
    print("  OpenAI Responses API - Compatibility Test Suite")
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {client.base_url}")
    print(f"  Response log: {os.path.abspath(LOG_FILE)}")
    print("=" * 60)

    _section("Basic Features")
    test_basic_response(client)
    test_instructions(client)
    test_multi_turn(client)
    test_reasoning_output(client)
    test_usage_fields(client)
    test_previous_response_id(client)
    test_retrieve_and_delete(client)

    _section("Streaming")
    test_streaming_basic(client)

    _section("Tool Use")
    test_single_tool_use(client)
    test_tool_result_roundtrip(client)
    test_mcp_tool_live_server(client)
    test_parallel_tool_use(client)

    _section("Structured Output")
    test_json_schema(client)

    _section("Multimodal")
    test_vision_input(client)

    print("\n" + "=" * 60)
    total = _pass + _fail + _skip
    print(f"  Results: {_pass} passed, {_fail} failed, {_skip} skipped / {total} total")
    print("=" * 60)

    log_path = _save_log()
    print(f"\n  All API responses saved to: {log_path}")
    print("  (use this file to debug failures)\n")
    return _fail == 0


if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LOCAL_BASE_URL", "http://0.0.0.0:8000/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy-key-for-local"),
    )

    success = run_all(client)
    raise SystemExit(0 if success else 1)
