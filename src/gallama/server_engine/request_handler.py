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


async def stream_response(method: str, url: str, headers: dict, body: bytes):   # TODO currently error while streaming was not returned to client
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(method, url, headers=headers, content=body, timeout=None) as response:
                if response.status_code >= 400:
                    error_message = json.dumps({"error": f"Server responded with status code: {response.status_code}"})
                    yield error_message.encode('utf-8')
                    return

                async for chunk in response.aiter_bytes():
                    yield chunk
    except httpx.RequestError as exc:
        error_message = json.dumps({"error": f"An error occurred: {str(exc)}"})
        yield error_message.encode('utf-8')
    except Exception as exc:
        error_message = json.dumps({"error": f"An unexpected error occurred: {str(exc)}"})
        yield error_message.encode('utf-8')


async def forward_request(
    request: Request,
    instance: ModelInstanceInfo,
    modified_body: Optional[str] = None,
    modified_headers: Optional[dict] = None
) -> Union[Response, StreamingResponse]:
    """
    Forward a request to a specific instance while handling optional modifications to the body and headers.

    Args:
        request (Request): The incoming HTTP request.
        instance (ModelInstanceInfo): Information about the target instance.
        modified_body (Optional[str]): Optional modified request body.
        modified_headers (Optional[dict]): Optional modified headers.

    Returns:
        Union[Response, StreamingResponse]: The forwarded response or a streaming response.
    """
    original_path = request.url.path
    path = original_path

    # Construct the URL to forward to
    url = f"http://localhost:{instance.port}{path}"
    logger.debug(f"Forwarding request to URL: {url}")

    # Use modified headers if provided, otherwise use the original headers
    headers = modified_headers if modified_headers else dict(request.headers)

    # Parse the request body
    body = modified_body if modified_body else await parse_request_body(request, return_full_body=True)
    body_json = None

    # Attempt to parse the body as JSON if applicable
    if isinstance(body, str):
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("Request body is not valid JSON; proceeding with raw body")

    request.state.instance_port = instance.port

    # Log relevant information
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Accept header: {headers.get('accept')}")
    logger.debug(f"Content-Type header: {headers.get('content-type')}")
    if body_json:
        logger.debug(f"Request body JSON: {body_json}")
    else:
        logger.debug(f"Request raw body: {body[:500]}")  # Log up to 500 characters of the raw body

    # Check if it's a streaming request
    is_streaming_request = body_json.get('stream', False) if body_json else False
    logger.debug(f"Is streaming request: {is_streaming_request}")

    if request.method == "OPTIONS":
        return create_options_response(headers)
    elif is_streaming_request:
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
                logger.info(f"Response status code: {response.status_code}")
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except httpx.RequestError as exc:
                logger.error(f"An error occurred while forwarding the request to instance at port {instance.port}: {exc}")
                raise HTTPException(status_code=500, detail="Internal server error")




