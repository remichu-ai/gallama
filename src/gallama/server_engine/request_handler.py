from gallama.logger import logger
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from gallama.data_classes import ModelInstanceInfo
from gallama.utils import parse_request_body
from typing import Union, Optional
import httpx


def create_options_response(headers: dict) -> Response:
    options_headers = {
        "Access-Control-Allow-Origin": headers.get("Origin", "*"),
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        "Access-Control-Max-Age": "3600",
    }
    return Response(content="", status_code=204, headers=options_headers)


async def stream_response(method: str, url: str, headers: dict, body: Union[str, dict, bytes, None]):
    """
    Stream a response from the target URL while handling different body types and errors.

    Args:
        method (str): HTTP method
        url (str): Target URL
        headers (dict): Request headers
        body: Request body (can be string, dict, bytes, or None)
    """
    try:
        # Convert body to appropriate format if needed
        if isinstance(body, dict):
            content = json.dumps(body).encode('utf-8')
        elif isinstance(body, str):
            content = body.encode('utf-8')
        elif isinstance(body, bytes):
            content = body
        else:
            content = None

        logger.debug(f"Streaming request to {url}")
        logger.debug(f"Body type: {type(body)}")
        logger.debug(f"Content type: {type(content)}")
        logger.debug(f"Content length: {len(content) if content else 0}")
        logger.debug(f"Headers: {headers}")

        async with httpx.AsyncClient() as client:
            async with client.stream(
                method,
                url,
                headers=headers,
                content=content,
                timeout=None
            ) as response:
                # Log response details
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")

                if response.status_code >= 400:
                    error_detail = await response.aread()
                    try:
                        error_json = json.loads(error_detail)
                        error_message = json.dumps({
                            "error": f"Server responded with status code: {response.status_code}",
                            "detail": error_json
                        })
                    except json.JSONDecodeError:
                        error_message = json.dumps({
                            "error": f"Server responded with status code: {response.status_code}",
                            "detail": error_detail.decode('utf-8', errors='replace')
                        })
                    yield error_message.encode('utf-8')
                    return

                async for chunk in response.aiter_bytes():
                    if chunk:  # Only yield non-empty chunks
                        logger.debug(f"Streaming chunk of size: {len(chunk)}")
                        yield chunk

    except httpx.RequestError as exc:
        logger.error(f"Request error while streaming: {exc}", exc_info=True)
        error_message = json.dumps({
            "error": "Request error",
            "detail": str(exc)
        })
        yield error_message.encode('utf-8')
    except Exception as exc:
        logger.error(f"Unexpected error while streaming: {exc}", exc_info=True)
        error_message = json.dumps({
            "error": "Unexpected error",
            "detail": str(exc)
        })
        yield error_message.encode('utf-8')


async def forward_request(
        request: Request,
        instance: ModelInstanceInfo,
        modified_body: Optional[str] = None,
        modified_headers: Optional[dict] = None
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

        # Handle the body based on content type and modifications
        if modified_body is not None:
            body = modified_body
            if isinstance(body, str):
                body = body.encode('utf-8')
        else:
            content_type = headers.get('content-type', '').lower()
            if 'application/json' in content_type:
                # For JSON, ensure we preserve the exact body
                if request._body:
                    body = request._body
                    try:
                        # Validate it's proper JSON but use original bytes
                        json.loads(body.decode('utf-8'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in request body: {e}")
                        raise HTTPException(status_code=400, detail="Invalid JSON")
                else:
                    body = b'{}'
            else:
                body = request._body

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

        # Check if it's a streaming request
        is_streaming = False
        if isinstance(body, bytes):
            try:
                body_json = json.loads(body.decode('utf-8'))
                is_streaming = body_json.get('stream', False)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if is_streaming:
            logger.info("Handling as streaming request")
            return StreamingResponse(
                stream_response(request.method, url, headers, body),
                media_type="application/json"
            )
        else:
            logger.info("Handling as non-streaming request")
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(
                        method=request.method,
                        url=url,
                        headers=headers,
                        content=body,
                        timeout=None
                    )
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except httpx.RequestError as exc:
                    logger.error(f"Request error: {exc}")
                    raise HTTPException(status_code=500, detail=str(exc))

    except Exception as e:
        logger.error(f"Error in forward_request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

