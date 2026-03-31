from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from mcp.types import LATEST_PROTOCOL_VERSION


DEFAULT_SERVER_LABEL = "dummy_mcp"
TEST_TOOL_NAME = "get_live_test_token"
TEST_TOKEN = "MCP_LIVE_TEST_TOKEN_4F7A1C8E"
TEST_USER_PROMPT = (
    "Use the MCP test tool to get the live test token and reply with only that token."
)


def build_openai_mcp_tool(
    server_url: str,
    *,
    server_label: str = DEFAULT_SERVER_LABEL,
) -> dict[str, Any]:
    return {
        "type": "mcp",
        "server_label": server_label,
        "server_url": server_url,
        "allowed_tools": [TEST_TOOL_NAME],
        "require_approval": "never",
    }


def build_responses_mcp_tool(
    server_url: str,
    *,
    server_label: str = DEFAULT_SERVER_LABEL,
) -> dict[str, Any]:
    return build_openai_mcp_tool(server_url, server_label=server_label)


def build_anthropic_mcp_server(
    server_url: str,
    *,
    server_label: str = DEFAULT_SERVER_LABEL,
) -> dict[str, Any]:
    return {
        "type": "url",
        "name": server_label,
        "url": server_url,
    }


def build_anthropic_mcp_toolset(
    *,
    server_label: str = DEFAULT_SERVER_LABEL,
) -> dict[str, Any]:
    return {
        "type": "mcp_toolset",
        "mcp_server_name": server_label,
        "allowed_tools": [TEST_TOOL_NAME],
    }


class _DummyMCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.request_log: list[dict[str, Any]] = []
        self.request_lock = threading.Lock()


class _DummyMCPHandler(BaseHTTPRequestHandler):
    server: _DummyMCPHTTPServer

    def _log(self, message: str) -> None:
        print(f"[dummy_mcp_server] {message}", flush=True)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int = 202) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        with self.server.request_lock:
            self.server.request_log.append(payload)

        method = payload.get("method")
        self._log(f"request path={self.path} method={method} payload={payload}")
        if method == "initialize":
            self._log("handling initialize")
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": LATEST_PROTOCOL_VERSION,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "dummy-mcp", "version": "1.0.0"},
                    },
                }
            )
            return

        if method == "notifications/initialized":
            self._log("handling notifications/initialized")
            self._send_empty()
            return

        if method == "tools/list":
            self._log("handling tools/list")
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": TEST_TOOL_NAME,
                                "description": (
                                    "Return the only valid token for the Gallama MCP live integration test."
                                ),
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                            }
                        ]
                    },
                }
            )
            return

        if method == "tools/call":
            params = payload.get("params", {})
            tool_name = params.get("name")
            self._log(f"handling tools/call tool={tool_name} arguments={params.get('arguments', {})}")
            if tool_name != TEST_TOOL_NAME:
                self._log(f"unknown tool requested: {tool_name}")
                self._send_json(
                    {
                        "jsonrpc": "2.0",
                        "id": payload.get("id"),
                        "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
                    },
                    status=400,
                )
                return

            self._log(f"returning token for tool={tool_name}")
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [{"type": "text", "text": TEST_TOKEN}],
                        "structuredContent": {
                            "token": TEST_TOKEN,
                            "source": "dummy_mcp_server",
                        },
                        "isError": False,
                    },
                }
            )
            return

        self._log(f"unknown method requested: {method}")
        self._send_json(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            },
            status=400,
        )

    def log_message(self, format: str, *args: Any) -> None:
        return None


@dataclass
class RunningDummyMCPServer:
    server: _DummyMCPHTTPServer
    thread: threading.Thread
    url: str

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self.server.request_lock:
            return list(self.server.request_log)

    def count_method(self, method: str) -> int:
        return sum(1 for payload in self.requests if payload.get("method") == method)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextmanager
def run_dummy_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Iterator[RunningDummyMCPServer]:
    server = _DummyMCPHTTPServer((host, port), _DummyMCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    running_server = RunningDummyMCPServer(
        server=server,
        thread=thread,
        url=f"http://{host}:{server.server_port}/mcp",
    )
    print(f"[dummy_mcp_server] listening on {running_server.url}", flush=True)
    try:
        yield running_server
    finally:
        print("[dummy_mcp_server] shutting down", flush=True)
        running_server.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a dummy MCP server for Gallama live tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    args = parser.parse_args()

    server = _DummyMCPHTTPServer((args.host, args.port), _DummyMCPHandler)
    url = f"http://{args.host}:{server.server_port}/mcp"
    print(url, flush=True)
    print(f"[dummy_mcp_server] listening on {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("[dummy_mcp_server] shutting down", flush=True)
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
