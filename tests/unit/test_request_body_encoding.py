import asyncio
import gzip
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import zstandard as zstd


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
        colorama_stub.Fore = types.SimpleNamespace(CYAN="", GREEN="", YELLOW="", RED="", BLUE="")
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

        class _HTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class _Request:
            pass

        fastapi_stub.FastAPI = object
        fastapi_stub.HTTPException = _HTTPException
        fastapi_stub.Query = lambda default=None, **kwargs: default
        fastapi_stub.Request = _Request
        sys.modules["fastapi"] = fastapi_stub

    if "fastapi.responses" not in sys.modules:
        fastapi_responses_stub = types.ModuleType("fastapi.responses")

        class _Response:
            def __init__(self, content=b"", status_code=200, headers=None, media_type=None):
                self.content = content
                self.status_code = status_code
                self.headers = headers or {}
                self.media_type = media_type

        class _StreamingResponse(_Response):
            pass

        fastapi_responses_stub.Response = _Response
        fastapi_responses_stub.StreamingResponse = _StreamingResponse
        sys.modules["fastapi.responses"] = fastapi_responses_stub


_install_test_stubs()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from gallama.utils.utils import decode_content_encoded_body, parse_request_body


REQUEST_HANDLER_PATH = os.path.join(SRC_DIR, "gallama", "server_engine", "request_handler.py")
REQUEST_HANDLER_SPEC = importlib.util.spec_from_file_location("request_handler_test_module", REQUEST_HANDLER_PATH)
REQUEST_HANDLER_MODULE = importlib.util.module_from_spec(REQUEST_HANDLER_SPEC)
assert REQUEST_HANDLER_SPEC.loader is not None
REQUEST_HANDLER_SPEC.loader.exec_module(REQUEST_HANDLER_MODULE)
forward_request = REQUEST_HANDLER_MODULE.forward_request
close_forward_http_client = REQUEST_HANDLER_MODULE.close_forward_http_client


class DummyRequest:
    def __init__(self, body: bytes, headers: dict[str, str], path: str = "/v1/responses"):
        self._initial_body = body
        self.headers = headers
        self.method = "POST"
        self.url = SimpleNamespace(path=path)
        self.state = SimpleNamespace()

    async def body(self):
        return self._initial_body


def test_decode_content_encoded_body_handles_zstd():
    payload = json.dumps({"model": "minimax", "input": "hello"}).encode("utf-8")
    encoded = zstd.ZstdCompressor().compress(payload)

    decoded = decode_content_encoded_body(encoded, "zstd")

    assert decoded == payload


def test_parse_request_body_handles_gzip_json():
    payload = json.dumps({"model": "minimax", "input": "hello"}).encode("utf-8")
    encoded = gzip.compress(payload)
    request = DummyRequest(
        body=encoded,
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        },
    )

    parsed_body, is_multipart = asyncio.run(parse_request_body(request))

    assert is_multipart is False
    assert parsed_body == {"model": "minimax", "input": "hello"}


def test_forward_request_decompresses_encoded_json_before_proxy(monkeypatch):
    payload = json.dumps({"model": "minimax", "input": "hello", "stream": False}).encode("utf-8")
    encoded = zstd.ZstdCompressor().compress(payload)
    request = DummyRequest(
        body=encoded,
        headers={
            "content-type": "application/json",
            "content-encoding": "zstd",
        },
    )
    captured = {}

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'
        headers = {"content-type": "application/json"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def aclose(self):
            return None

        async def request(self, method, url, headers, content, timeout=None):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers
            captured["content"] = content
            return FakeResponse()

    asyncio.run(close_forward_http_client())
    monkeypatch.setattr(REQUEST_HANDLER_MODULE.httpx, "AsyncClient", FakeAsyncClient)

    response = asyncio.run(
        forward_request(
            request,
            SimpleNamespace(port=8001),
            parsed_body={"model": "minimax", "input": "hello", "stream": False},
        )
    )
    asyncio.run(close_forward_http_client())

    assert response.status_code == 200
    assert captured["url"] == "http://localhost:8001/v1/responses"
    assert captured["content"] == payload
    assert "content-encoding" not in captured["headers"]
    assert captured["headers"]["content-length"] == str(len(payload))


def test_forward_request_reuses_shared_async_client(monkeypatch):
    payload = json.dumps({"model": "minimax", "input": "hello", "stream": False}).encode("utf-8")
    requests_seen = []
    client_count = 0

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'
        headers = {"content-type": "application/json"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            nonlocal client_count
            client_count += 1

        async def aclose(self):
            return None

        async def request(self, method, url, headers, content, timeout=None):
            requests_seen.append((method, url, content))
            return FakeResponse()

    asyncio.run(close_forward_http_client())
    monkeypatch.setattr(REQUEST_HANDLER_MODULE.httpx, "AsyncClient", FakeAsyncClient)

    first_request = DummyRequest(
        body=payload,
        headers={"content-type": "application/json"},
    )
    second_request = DummyRequest(
        body=payload,
        headers={"content-type": "application/json"},
    )

    asyncio.run(
        forward_request(
            first_request,
            SimpleNamespace(port=8001),
            parsed_body={"model": "minimax", "input": "hello", "stream": False},
        )
    )
    asyncio.run(
        forward_request(
            second_request,
            SimpleNamespace(port=8002),
            parsed_body={"model": "minimax", "input": "hello", "stream": False},
        )
    )
    asyncio.run(close_forward_http_client())

    assert client_count == 1
    assert requests_seen == [
        ("POST", "http://localhost:8001/v1/responses", payload),
        ("POST", "http://localhost:8002/v1/responses", payload),
    ]


def test_forward_request_uses_preparsed_body_without_json_reparse(monkeypatch):
    payload = json.dumps({"model": "minimax", "input": "hello", "stream": False}).encode("utf-8")
    request = DummyRequest(
        body=payload,
        headers={"content-type": "application/json"},
    )

    class FakeResponse:
        status_code = 200
        content = b'{"ok":true}'
        headers = {"content-type": "application/json"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def aclose(self):
            return None

        async def request(self, method, url, headers, content, timeout=None):
            return FakeResponse()

    def fail_json_loads(*args, **kwargs):
        raise AssertionError("json.loads should not run when parsed_body is already provided")

    asyncio.run(close_forward_http_client())
    monkeypatch.setattr(REQUEST_HANDLER_MODULE.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(REQUEST_HANDLER_MODULE.json, "loads", fail_json_loads)

    response = asyncio.run(
        forward_request(
            request,
            SimpleNamespace(port=8001),
            parsed_body={"model": "minimax", "input": "hello", "stream": False},
        )
    )
    asyncio.run(close_forward_http_client())

    assert response.status_code == 200
