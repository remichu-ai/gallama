from types import SimpleNamespace

import gallama.server as server


class _CaptureLogger:
    def __init__(self):
        self.debugs = []
        self.infos = []

    def debug(self, message, **kwargs):
        self.debugs.append((message, kwargs))

    def info(self, message, **kwargs):
        self.infos.append((message, kwargs))


def test_request_received_log_is_debug_only(monkeypatch):
    logger = _CaptureLogger()
    monkeypatch.setattr(server, "server_logger", logger)
    request = SimpleNamespace(
        state=SimpleNamespace(request_id="abcd1234"),
        method="POST",
        url=SimpleNamespace(path="/v1/messages"),
    )

    server._log_request_received(request)

    assert logger.debugs == [("REQ abcd1234 POST /v1/messages", {})]
    assert logger.infos == []
