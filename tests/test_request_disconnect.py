import asyncio
import importlib.util
import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_PATH = os.path.join(ROOT_DIR, "src", "gallama", "utils", "request_disconnect.py")
MODULE_SPEC = importlib.util.spec_from_file_location("request_disconnect_test_module", MODULE_PATH)
REQUEST_DISCONNECT_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(REQUEST_DISCONNECT_MODULE)

is_expected_disconnect_exception = REQUEST_DISCONNECT_MODULE.is_expected_disconnect_exception
is_request_disconnected = REQUEST_DISCONNECT_MODULE.is_request_disconnected


class DummyRequest:
    def __init__(self, receive, is_disconnected: bool = False):
        self.receive = receive
        self._is_disconnected = is_disconnected


def test_is_request_disconnected_marks_http_disconnect():
    async def receive():
        return {"type": "http.disconnect"}

    request = DummyRequest(receive)

    assert asyncio.run(is_request_disconnected(request)) is True
    assert request._is_disconnected is True


def test_is_request_disconnected_returns_false_when_no_message_is_ready():
    async def receive():
        await asyncio.sleep(1)
        return {"type": "http.request"}

    request = DummyRequest(receive)

    assert asyncio.run(is_request_disconnected(request, probe_timeout=0.001)) is False
    assert request._is_disconnected is False


def test_is_request_disconnected_treats_assertion_error_as_disconnect():
    async def receive():
        raise AssertionError()

    request = DummyRequest(receive)

    assert asyncio.run(is_request_disconnected(request)) is True
    assert request._is_disconnected is True


def test_is_expected_disconnect_exception_accepts_client_disconnect_name():
    class ClientDisconnect(Exception):
        pass

    assert is_expected_disconnect_exception(ClientDisconnect()) is True
