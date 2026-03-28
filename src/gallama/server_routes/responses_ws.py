import contextlib
import asyncio
import json
from typing import Optional

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies_server import get_responses_websocket_hub, get_server_logger, get_server_manager
from ..server_engine.responses_ws_bridge import (
    extract_response_ws_keys,
    extract_responses_request_payload,
    stream_responses_request_to_events,
)


router = APIRouter(prefix="", tags=["responses_ws"])

server_manager = get_server_manager()
responses_websocket_hub = get_responses_websocket_hub()
logger = get_server_logger()


def _select_llm_instance(model_name: Optional[str]):
    if model_name and model_name in server_manager.models:
        for instance in server_manager.models[model_name].instances:
            if instance.status == "running":
                return instance

    return server_manager.get_instance(model_type="llm", model_name=model_name)


@router.websocket("/v1/responses")
async def responses_websocket_endpoint(websocket: WebSocket):
    headers = dict(websocket.headers)
    keys = extract_response_ws_keys(headers)
    active_stream_task: Optional[asyncio.Task] = None

    await websocket.accept()
    await responses_websocket_hub.register(websocket, keys)

    try:
        while True:
            raw_message = await websocket.receive_text()
            if not raw_message:
                continue

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON /v1/responses websocket message")
                continue

            if payload.get("type") in {"ping", "websocket.ping"}:
                await websocket.send_json({"type": "pong"})
                continue

            request_payload = extract_responses_request_payload(payload)
            if request_payload is None:
                logger.debug("Ignoring /v1/responses websocket payload without a request body")
                continue

            instance = _select_llm_instance(request_payload.get("model"))
            if instance is None:
                await websocket.send_json(
                    {
                        "type": "response.failed",
                        "error": {
                            "message": f"No running LLM instance available for model '{request_payload.get('model')}'",
                        },
                    }
                )
                continue

            if active_stream_task is not None and not active_stream_task.done():
                active_stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await active_stream_task

            async def _forward_stream():
                async def on_event(_event_name, event_payload):
                    if keys:
                        await responses_websocket_hub.publish(keys, event_payload)
                    else:
                        await websocket.send_json(event_payload)

                try:
                    await stream_responses_request_to_events(
                        instance=instance,
                        payload=request_payload,
                        headers=headers,
                        on_event=on_event,
                    )
                except httpx.HTTPStatusError as exc:
                    await websocket.send_json(
                        {
                            "type": "response.failed",
                            "error": {
                                "message": exc.response.text,
                                "status_code": exc.response.status_code,
                            },
                        }
                    )

            active_stream_task = asyncio.create_task(_forward_stream())
            await active_stream_task

    except WebSocketDisconnect:
        logger.debug("Responses websocket disconnected")
    finally:
        if active_stream_task is not None and not active_stream_task.done():
            active_stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active_stream_task
        await responses_websocket_hub.unregister(websocket)
