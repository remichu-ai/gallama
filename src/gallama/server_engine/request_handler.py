from gallama.logger import logger
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from gallama.data_classes import ModelInstanceInfo
from typing import Union
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


async def forward_request(request: Request, instance: ModelInstanceInfo, modified_body: str = None, modified_headers: str = None) -> Union[Response, StreamingResponse]:
    original_path = request.url.path
    path = original_path

    # # Check if the path matches any in EMBEDDING_SUBPATHS and replace if necessary
    # for subpath in EMBEDDING_SUBPATHS:
    #     if path.startswith(subpath["original"]):
    #         path = path.replace(subpath["original"], subpath["replacement"], 1)
    #         logger.info(f"Path replaced: {original_path} -> {path}")
    #         break

    url = f"http://localhost:{instance.port}{path}"
    logger.debug(f"Forwarding request to URL: {url}")

    headers = modified_headers if modified_headers is not None else dict(request.headers)
    body = modified_body if modified_body is not None else await request.body()
    body_json = json.loads(body)

    request.state.instance_port = instance.port

    # Log relevant information
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Accept header: {headers.get('accept')}")
    logger.debug(f"Content-Type header: {headers.get('content-type')}")
    logger.debug(f"Request body: {body_json}")

    # Check if it's a streaming request
    is_streaming_request = body_json.get('stream', False)
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
                response = await client.request(method=request.method, url=url, headers=headers, content=body, timeout=None)
                logger.info(f"Response status code: {response.status_code}")
                return Response(content=response.content, status_code=response.status_code,
                                headers=dict(response.headers))
            except httpx.RequestError as exc:
                logger.error(f"An error occurred while forwarding the request to instance at port {instance.port}: {exc}")
                raise HTTPException(status_code=500, detail="Internal server_engine error")



