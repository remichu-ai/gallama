import asyncio
import importlib.util
import json
import os
import sys
import types

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

if "colorama" not in sys.modules:
    colorama_stub = types.ModuleType("colorama")
    colorama_stub.Fore = types.SimpleNamespace(CYAN="", GREEN="", YELLOW="", RED="", BLUE="")
    colorama_stub.Back = types.SimpleNamespace(WHITE="")
    colorama_stub.Style = types.SimpleNamespace(RESET_ALL="")
    colorama_stub.init = lambda autoreset=True: None
    sys.modules["colorama"] = colorama_stub

fastapi_stub = sys.modules.get("fastapi")
if fastapi_stub is None:
    fastapi_stub = types.ModuleType("fastapi")
    sys.modules["fastapi"] = fastapi_stub
if not hasattr(fastapi_stub, "WebSocket"):
    fastapi_stub.WebSocket = object
if not hasattr(fastapi_stub, "Query"):
    fastapi_stub.Query = lambda default=None, **kwargs: default

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def device_count():
            return 0

    torch_stub.cuda = _Cuda()
    sys.modules["torch"] = torch_stub

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

MODULE_PATH = os.path.join(SRC_DIR, "gallama", "server_engine", "responses_ws_bridge.py")
MODULE_SPEC = importlib.util.spec_from_file_location("responses_ws_bridge_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(MODULE)

extract_response_ws_keys = MODULE.extract_response_ws_keys
extract_responses_request_payload = MODULE.extract_responses_request_payload
iter_sse_events = MODULE.iter_sse_events


class _FakeSSEStreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_text(self):
        for chunk in self._chunks:
            yield chunk


def test_extract_response_ws_keys_reads_codex_headers():
    headers = {
        "session_id": "sess-123",
        "x-client-request-id": "client-456",
        "x-codex-turn-metadata": json.dumps(
            {
                "session_id": "sess-123",
                "turn_id": "turn-789",
            }
        ),
    }

    keys = extract_response_ws_keys(headers)

    assert "session:sess-123" in keys
    assert "turn:turn-789" in keys
    assert "session-turn:sess-123:turn-789" in keys


def test_extract_responses_request_payload_accepts_wrapped_request():
    payload = {
        "type": "request.create",
        "request": {
            "model": "minimax",
            "input": "hello",
        },
    }

    extracted_payload = extract_responses_request_payload(payload)

    assert extracted_payload == {"model": "minimax", "input": "hello"}


def test_iter_sse_events_parses_chunked_worker_stream():
    raw_chunks = [
        'event: response.created\r\ndata: {"type":"response.created","response":{"id":"resp_1"}}\r\n\r\n',
        'event: response.completed\r\ndata: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}\r\n\r\n',
    ]

    async def collect_events():
        return [event async for event in iter_sse_events(_FakeSSEStreamResponse(raw_chunks))]

    events = asyncio.run(collect_events())

    assert events == [
        ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
        ("response.completed", {"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}}),
    ]
