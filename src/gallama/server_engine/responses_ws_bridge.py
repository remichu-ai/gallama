import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Deque, Dict, Mapping, Optional, Set, Tuple

import httpx
from fastapi import WebSocket

from gallama.data_classes import ModelInstanceInfo
from gallama.logger import logger


def extract_response_ws_keys(headers: Mapping[str, str]) -> Set[str]:
    normalized = {str(key).lower(): value for key, value in headers.items()}
    keys: Set[str] = set()

    session_id = normalized.get("session_id") or normalized.get("x-client-request-id")
    turn_id: Optional[str] = None

    raw_turn_metadata = normalized.get("x-codex-turn-metadata")
    if raw_turn_metadata:
        try:
            parsed_turn_metadata = json.loads(raw_turn_metadata)
        except json.JSONDecodeError:
            logger.debug("Unable to parse x-codex-turn-metadata as JSON")
        else:
            metadata_session_id = parsed_turn_metadata.get("session_id")
            metadata_turn_id = parsed_turn_metadata.get("turn_id")
            if metadata_session_id and not session_id:
                session_id = metadata_session_id
            if metadata_turn_id:
                turn_id = metadata_turn_id

    if session_id:
        keys.add(f"session:{session_id}")
    if turn_id:
        keys.add(f"turn:{turn_id}")
    if session_id and turn_id:
        keys.add(f"session-turn:{session_id}:{turn_id}")

    return keys


def is_codex_responses_transport(headers: Mapping[str, str]) -> bool:
    normalized = {str(key).lower(): value for key, value in headers.items()}
    beta_header = normalized.get("openai-beta", "")
    originator = normalized.get("originator", "")
    return (
        "responses_websockets=" in beta_header
        or originator == "codex_cli_rs"
        or bool(extract_response_ws_keys(normalized))
    )


def extract_responses_request_payload(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return None

    if "model" in message and "input" in message:
        return message

    for key in ("request", "response", "params", "payload"):
        value = message.get(key)
        if isinstance(value, dict) and "model" in value and "input" in value:
            return value

    return None


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue

        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            parsed = _parse_sse_event(raw_event)
            if parsed is not None:
                yield parsed

    parsed = _parse_sse_event(buffer)
    if parsed is not None:
        yield parsed


def _parse_sse_event(raw_event: str) -> Optional[Tuple[Optional[str], Optional[Dict[str, Any]]]]:
    raw_event = raw_event.strip()
    if not raw_event:
        return None

    event_name: Optional[str] = None
    data_lines = []
    for line in raw_event.split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())

    if not data_lines:
        return event_name, None

    payload_raw = "\n".join(data_lines)
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        logger.debug("Unable to parse SSE payload as JSON: %s", payload_raw)
        return event_name, None

    return event_name, payload


@dataclass
class ManagedResponsesWebSocket:
    websocket: WebSocket
    keys: Set[str]
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ResponsesWebSocketHub:
    def __init__(self, history_limit: int = 64):
        self._connections: Dict[int, ManagedResponsesWebSocket] = {}
        self._history: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=history_limit))
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket, keys: Set[str]):
        managed_socket = ManagedResponsesWebSocket(websocket=websocket, keys=set(keys))

        async with self._lock:
            self._connections[id(websocket)] = managed_socket
            history_to_replay = self._collect_history_for_keys(keys)

        for serialized_payload in history_to_replay:
            await self._send_serialized(managed_socket, serialized_payload)

    async def unregister(self, websocket: WebSocket):
        async with self._lock:
            self._connections.pop(id(websocket), None)

    async def publish(self, keys: Set[str], payload: Dict[str, Any]):
        if not keys:
            return

        serialized_payload = json.dumps(payload, ensure_ascii=False)

        async with self._lock:
            for key in keys:
                self._history[key].append(serialized_payload)

            recipients = [
                managed_socket
                for managed_socket in self._connections.values()
                if managed_socket.keys.intersection(keys)
            ]

        stale_socket_ids = []
        for managed_socket in recipients:
            try:
                await self._send_serialized(managed_socket, serialized_payload)
            except Exception:
                stale_socket_ids.append(id(managed_socket.websocket))

        if stale_socket_ids:
            async with self._lock:
                for socket_id in stale_socket_ids:
                    self._connections.pop(socket_id, None)

    def has_subscribers(self, keys: Set[str]) -> bool:
        if not keys:
            return False
        return any(
            managed_socket.keys.intersection(keys)
            for managed_socket in self._connections.values()
        )

    def _collect_history_for_keys(self, keys: Set[str]) -> list[str]:
        seen_payloads = set()
        replay_payloads = []
        for key in keys:
            for serialized_payload in self._history.get(key, ()):
                if serialized_payload in seen_payloads:
                    continue
                seen_payloads.add(serialized_payload)
                replay_payloads.append(serialized_payload)
        return replay_payloads

    async def _send_serialized(self, managed_socket: ManagedResponsesWebSocket, serialized_payload: str):
        async with managed_socket.send_lock:
            await managed_socket.websocket.send_text(serialized_payload)


async def stream_responses_request_to_events(
    instance: ModelInstanceInfo,
    payload: Dict[str, Any],
    headers: Optional[Mapping[str, str]] = None,
    on_event: Optional[Callable[[Optional[str], Dict[str, Any]], Awaitable[None]]] = None,
) -> Optional[Dict[str, Any]]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    url = f"http://localhost:{instance.port}/v1/responses"
    request_headers = _prepare_backend_headers(headers)

    final_response: Optional[Dict[str, Any]] = None

    async with httpx.AsyncClient(timeout=None) as client:
        request = client.build_request(
            method="POST",
            url=url,
            headers=request_headers,
            content=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
        )
        response = await client.send(request, stream=True)
        if response.status_code >= 400:
            error_body = await response.aread()
            raise httpx.HTTPStatusError(
                f"Backend returned {response.status_code}: {error_body.decode('utf-8', errors='replace')}",
                request=request,
                response=response,
            )

        async for event_name, event_payload in iter_sse_events(response):
            if not event_payload:
                continue

            if event_payload.get("type") == "response.completed":
                final_response = event_payload.get("response")

            if on_event is not None:
                await on_event(event_name, event_payload)

        await response.aclose()

    return final_response


def _prepare_backend_headers(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    filtered_headers = {
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if headers is None:
        return filtered_headers

    blocked_headers = {
        "accept",
        "connection",
        "content-encoding",
        "content-length",
        "host",
        "upgrade",
    }
    for key, value in headers.items():
        normalized_key = key.lower()
        if normalized_key.startswith("sec-websocket-") or normalized_key in blocked_headers:
            continue
        filtered_headers[normalized_key] = value

    filtered_headers["accept"] = "text/event-stream"
    filtered_headers["content-type"] = "application/json"
    return filtered_headers
