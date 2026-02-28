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

            # 1. Prepare the client and request with NO TIMEOUT
            client = httpx.AsyncClient(timeout=None)
            req = client.build_request(
                method=request.method,
                url=url,
                headers=headers,
                content=body
            )

            try:
                # 2. Send the request (this will now wait patiently for the LLM's first token)
                response = await client.send(req, stream=True)
            except Exception as e:
                # Clean up the client if the server completely drops the connection
                await client.aclose()
                logger.error(f"Failed to connect or receive headers from target: {e}")
                raise HTTPException(status_code=502, detail="Bad Gateway: Target server failed to respond.")

            # 3. Catch errors immediately to prevent FastAPI from returning a 200 OK
            if response.status_code >= 400:
                await response.aread()
                error_detail = response.text
                await response.aclose()
                await client.aclose()

                logger.error(f"Backend returned {response.status_code}: {error_detail}")

                # Raise the exact status code to the client
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Target server error: {error_detail}"
                )

            # 4. If successful, set up the generator
            async def stream_generator():
                try:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
                finally:
                    # Ensure connections are cleanly closed when streaming ends or disconnects
                    await response.aclose()
                    await client.aclose()

            # 5. Return StreamingResponse with the correct HTTP status code
            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="text/event-stream"
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

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI handles them correctly instead of wrapping them in a 500
        raise
    except Exception as e:
        logger.error(f"Error in forward_request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))