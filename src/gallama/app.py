# using absolute import here as this file will be run alone
import sys
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from gallama.data_classes import (
    ModelSpec,
    GenQueue,
    GenQueueDynamic
)
import signal
import argparse
import uvicorn
from fastapi.exceptions import RequestValidationError
import json
from fastapi.responses import JSONResponse, StreamingResponse
from gallama.utils import parse_request_body
from gallama.config.config_manager import ConfigManager
import os
from contextlib import asynccontextmanager
from gallama.logger.logger import (
    REQUEST_ID_HEADER,
    basic_log_extra,
    get_logger,
    get_log_verbosity,
    get_log_level_for_verbosity,
    new_request_id,
    reset_request_id,
    set_log_verbosity,
    set_request_id,
)
import base64
from gallama.dependencies import get_model_manager
from gallama.warmup import warmup_llm
from gallama.routes import (
    chat_router,
    embedding_router,
    model_management_router,
    audio_router,
    ws_stt_router,
    ws_llm_router,
    ws_tts_router,
    ws_video_router
)
from tempfile import SpooledTemporaryFile

# Add this after your imports to clear logging from 3rd party module

#streaming example
#https://blog.gopenai.com/how-to-stream-output-in-llm-based-applications-to-boost-user-experience-e9fcf582777a

logger = get_logger()

# add endpoint
router = APIRouter()
router.include_router(chat_router)
router.include_router(embedding_router)
router.include_router(model_management_router)
router.include_router(audio_router)
router.include_router(ws_stt_router)
router.include_router(ws_llm_router)
router.include_router(ws_tts_router)
router.include_router(ws_video_router)


config_manager = ConfigManager()
model_manager = get_model_manager()
model_ready = False

@router.get("/")
async def read_root():
    return {"Hello": "PAI"}


@router.options("/v1/chat/completions")
@router.options("/v1/responses")
@router.options("/v1/conversations")
@router.options("/v1/conversations/{conversation_id}")
async def options_handler(request: Request):
    return JSONResponse(
        status_code=200,
        content={"message": "Local OpenAI response for OPTIONS request"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "",
        },
    )



async def startup_event():
    # run some dummy generation so that cache is initialized
    logger.info("Generator initialization", extra=basic_log_extra())

    # warm up LLM
    if model_manager.llm_dict:
        gen_queue = GenQueue()
        warmup_base_dir = config_manager.get_gallama_user_config_file_path.parent
        for _model_name, _model in model_manager.llm_dict.items():
            await warmup_llm(
                model=_model,
                model_name=_model_name,
                warmup_prompt=getattr(_model, "warmup_prompt", None),
                gen_queue=gen_queue,
                base_dir=warmup_base_dir,
            )
            logger.info(f"LLM| {_model_name} | warmed up", extra=basic_log_extra())
        gen_queue = None

    if model_manager.stt_dict:
        logger.info("STT warmed up NOT YET IMPLEMENTED", extra=basic_log_extra())

    if model_manager.tts_dict:
        for _model_name, tts in model_manager.tts_dict.items():
            _, _ = await tts.text_to_speech(
                text="hello",
                stream=False,
                batching=False,
                batch_size=1
            )
            logger.info(f"TTS| {_model_name} | warmed up", extra=basic_log_extra())

    if model_manager.embedding_dict:
        logger.info("Embedding warmed up NOT YET IMPLEMENTED", extra=basic_log_extra())



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await startup_event()
    try:
        yield
    finally:
        # Cleanup code
        logger.info("Cleaning up loaded models...", extra=basic_log_extra())
        model_manager.close_all_models()
        logger.info("Cleaning up ZMQ connections...", extra=basic_log_extra())
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()


def make_server(args):
    global logger
    global draft_spec_dict
    requested_verbosity = getattr(args, "verbose", 0) or get_log_verbosity()
    set_log_verbosity(requested_verbosity)

    logger = get_logger(
        log_file=args.log_file or "./log/llm_response.log",
        to_console=True,
        to_file=bool(args.log_file),
        to_zmq=False
    )

    # Add signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...", extra=basic_log_extra())
        model_manager.close_all_models()
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # load yaml file of model info
    logger.info(args, extra=basic_log_extra())
    model_dict = {}
    draft_spec_dict = {}    # for speculative decoding

    # parse cli argument to get model info
    # Handle model specification
    model_spec = None
    if args.model_spec:
        try:
            # Decode and parse the ModelSpec from base64 JSON
            model_json = base64.b64decode(args.model_spec).decode('utf-8')
            model_spec = ModelSpec.model_validate_json(model_json)
        except Exception as e:
            logger.error(f"Failed to parse model specification: {e}")
            sys.exit(1)

        # set the env for log to zmq
        # Note: This assumes you want to use the first model's ID if multiple are provided
        os.environ['MODEL_NAME'] = model_spec.model_name
        os.environ['MODEL_PORT'] = str(args.port)

    logger.setLevel(get_log_level_for_verbosity())

    args.model_spec = model_spec
    logger.info("Parsed Arguments:" + str(args), extra=basic_log_extra())

    if args.detached:
        # Reconfigure the shared package logger used across imported modules so
        # worker logs are forwarded back to the manager via ZMQ.
        logger = get_logger(name="logger", to_zmq=True, to_console=False)
    else:
        # keep the default logger declared on top
        pass

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], #origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.middleware("http")
    @app.middleware("https")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        request.state.request_id = request_id
        token = set_request_id(request_id)
        try:
            # if request.method in ("POST", "PUT", "PATCH"):
            #     # Parse body and preserve it
            #     body_content, is_multipart = await parse_request_body(request)
            #
            #     # Handle file uploads explicitly
            #     if is_multipart or isinstance(body_content, SpooledTemporaryFile):
            #         request_content = "File upload content not logged"
            #     elif isinstance(body_content, dict):
            #         request_content = json.dumps(body_content, indent=2)
            #     elif isinstance(body_content, str):
            #         try:
            #             parsed_content = json.loads(body_content)
            #             request_content = json.dumps(parsed_content, indent=2)
            #         except json.JSONDecodeError:
            #             request_content = body_content
            #     else:
            #         request_content = str(body_content)
            #
            #     logger.debug(
            #         f"API Request:\n"
            #         f"Method: {request.method}\n"
            #         f"URL: {request.url}\n"
            #         f"Headers: {dict(request.headers)}\n"
            #         f"Content-Type: {request.headers.get('content-type', 'Not specified')}\n"
            #         f"Body: {request_content if request_content else 'No Body'}"
            #     )

            # Proceed with the request
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            if isinstance(response, StreamingResponse):
                original_iterator = response.body_iterator

                async def context_body_iterator():
                    stream_token = set_request_id(request_id)
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    finally:
                        reset_request_id(stream_token)

                response.body_iterator = context_body_iterator()
            return response

        except Exception as e:
            logger.error(f"Middleware error: {str(e)}", exc_info=True)
            error_response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error in middleware"}
            )
            error_response.headers[REQUEST_ID_HEADER] = request_id
            return error_response
        finally:
            reset_request_id(token)

    if model_spec:
        # load model
        model_manager.load_model(model_spec)
        model_manager.model_ready = True

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)    # disable log_config to use our custom logger


def parse_dict(arg):
    """Parses key=value pairs and supports dotted keys for nested dictionaries."""

    def assign_nested_key(target, dotted_key, value):
        parts = dotted_key.split(".")
        current = target
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    result = {}
    for pair in arg.split():
        key, value = pair.split('=', 1)
        assign_nested_key(result, key, value.strip("'"))
    return result


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Launch local AI model")
    arg_parser.add_argument("--model-spec", type=str, help="Base64 encoded JSON ModelSpec object")
    arg_parser.add_argument(
        '-v',
        "--verbose",
        action='count',
        default=0,
        help="Increase logging verbosity. Use -v for current logs, -vv for debug, -vvv for maximum request/body detail.",
    )
    arg_parser.add_argument('-d', "--detached", action='store_true', help="Log to ZeroMQ")
    arg_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
    arg_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")
    arg_parser.add_argument("--log-file", type=str, default=None, help="Also write CLI logs to this file.")
    arg_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    args = arg_parser.parse_args()

    make_server(args)
