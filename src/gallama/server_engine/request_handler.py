import asyncio
import json
from typing import Optional, Union

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from gallama.data_classes import ModelInstanceInfo
from gallama.logger import logger
from gallama.utils.utils import decode_content_encoded_body


_forward_http_client: httpx.AsyncClient | None = None
_forward_http_client_lock = asyncio.Lock()


async def get_forward_http_client() -> httpx.AsyncClient:
    global _forward_http_client

    if _forward_http_client is not None:
        return _forward_http_client

    async with _forward_http_client_lock:
        if _forward_http_client is None:
            _forward_http_client = httpx.AsyncClient(timeout=None)

    return _forward_http_client


async def close_forward_http_client() -> None:
    global _forward_http_client

    client = _forward_http_client
    if client is None:
        return

    _forward_http_client = None
    await client.aclose()


def create_options_response(headers: dict) -> Response:
    options_headers = {
        "Access-Control-Allow-Origin": headers.get("Origin", "*"),
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        "Access-Control-Max-Age": "3600",
    }
    return Response(content="", status_code=204, headers=options_headers)


async def forward_request(
        request: Request,
        instance: ModelInstanceInfo,
        modified_body: Optional[str] = None,
        modified_headers: Optional[dict] = None,
        parsed_body: Optional[Union[dict, str, bytes]] = None,
) -> Union[Response, StreamingResponse]:
    """
    Forward a request to a specific instance while handling optional modifications to the body and headers.
    """
    try:
        original_path = request.url.path
        url = f"http://localhost:{instance.port}{original_path}"
        logger.debug(f"Forwarding request to URL: {url}")

        # Use modified headers if provided, otherwise use the original headers
        headers = modified_headers if modified_headers else dict(request.headers)

        # Ensure we have the raw body bytes first
        if not hasattr(request, '_body'):
            request._body = await request.body()

            async def get_body():
                return request._body

            request.body = get_body

        is_streaming = False

        # Handle the body based on content type and modifications
        if modified_body is not None:
            body = modified_body
            if isinstance(body, str):
                body = body.encode('utf-8')
        else:
            content_type = headers.get('content-type', '').lower()
            content_encoding = headers.get('content-encoding', '')
            if request._body and content_encoding:
                body = decode_content_encoded_body(request._body, content_encoding)
                headers.pop('content-encoding', None)
            else:
                body = request._body

            if 'application/json' in content_type:
                if not body:
                    body = b'{}'
                    parsed_body = {}
                elif isinstance(parsed_body, dict):
                    is_streaming = bool(parsed_body.get("stream", False))
                elif parsed_body is not None:
                    logger.error("Invalid JSON in request body")
                    raise HTTPException(status_code=400, detail="Invalid JSON")
                else:
                    try:
                        parsed_body = json.loads(body.decode('utf-8'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in request body: {e}")
                        raise HTTPException(status_code=400, detail="Invalid JSON")

                    if isinstance(parsed_body, dict):
                        is_streaming = bool(parsed_body.get("stream", False))

        # Update Content-Length header to match actual body length
        if body:
            headers['content-length'] = str(len(body))

        # Log the request details
        logger.debug(f"Forwarding request:")
        logger.debug(f"Method: {request.method}")
        logger.debug(f"URL: {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Body type: {type(body)}")
        logger.debug(f"Body length: {len(body) if body else 0}")
        if isinstance(body, bytes):
            try:
                logger.debug(f"Body content: {body.decode('utf-8')[:500]}")
            except UnicodeDecodeError:
                logger.debug("Body contains binary data")

        request.state.instance_port = instance.port

        if request.method == "OPTIONS":
            return create_options_response(headers)

        if is_streaming:
            logger.info("Handling as streaming request")

            client = await get_forward_http_client()
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body
            )

            try:
                response = await client.send(req, stream=True)
            except Exception as e:
                logger.error(f"Failed to connect or receive headers from target: {e}")
                raise HTTPException(status_code=502, detail="Bad Gateway: Target server failed to respond.")

            if response.status_code >= 400:
                await response.aread()
                error_detail = response.text
                await response.aclose()

                logger.error(f"Backend returned {response.status_code}: {error_detail}")

                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Target server error: {error_detail}"
                )

            async def stream_generator():
                try:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
                finally:
                    await response.aclose()

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="text/event-stream"
            )

        else:
            logger.info("Handling as non-streaming request")
            client = await get_forward_http_client()
            try:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                )
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except httpx.RequestError as exc:
                logger.error(f"Request error: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI handles them correctly instead of wrapping them in a 500
        raise
    except Exception as e:
        logger.error(f"Error in forward_request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
