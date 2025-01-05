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
from fastapi.responses import JSONResponse
from gallama.utils import parse_request_body
from gallama.config.config_manager import ConfigManager
from gallama.data import ARTIFACT_SYSTEM_PROMPT
import os
from contextlib import asynccontextmanager
from logging import DEBUG
from gallama.logger.logger import get_logger
import base64
from gallama.dependencies import get_model_manager
from gallama.routes import (
    chat_router,
    embedding_router,
    model_management_router,
    audio_router,
    ws_stt_router,
    ws_llm_router,
    ws_tts_router
)
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


config_manager = ConfigManager()
model_manager = get_model_manager()
model_ready = False

@router.get("/")
async def read_root():
    return {"Hello": "PAI"}


@router.options("/v1/chat/completions")
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
    logger.info("Generator initialization")

    # warm up LLM
    if model_manager.llm_dict:
        gen_queue = GenQueue()
        for _model_name, _model in model_manager.llm_dict.items():
            await _model.chat_raw(
                prompt=f"{ARTIFACT_SYSTEM_PROMPT}\nWrite a 500 words story on Llama",
                # stream=False,
                max_tokens=100,
                gen_queue=gen_queue,
                quiet=True,
                request=None,
            )

            logger.info(f"LLM| {_model_name} | warmed up")
        gen_queue = None

    if model_manager.stt_dict:
        logger.info("STT warmed up NOT YET IMPLEMENTED")

    if model_manager.tts_dict:
        for _model_name, tts in model_manager.tts_dict.items():
            _, _ = await tts.text_to_speech(
                text="hello",
                stream=False,
                batching=False,
                batch_size=1
            )
            logger.info(f"TTS| {_model_name} | warmed up")

    if model_manager.embedding_dict:
        logger.info("Embedding warmed up NOT YET IMPLEMENTED")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await startup_event()
    try:
        yield
    finally:
        # Cleanup code
        logger.info("Cleaning up ZMQ connections...")
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()


def make_server(args):
    global logger
    global draft_spec_dict

    # Add signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        for handler in logger.handlers:
            if hasattr(handler, 'close'):
                handler.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # load yaml file of model info
    logger.info(args)
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

    # set logger level
    if args.verbose:
        logger.setLevel(DEBUG)
        os.environ["LOCAL_OPEN_AI_VERBOSE"] = '2'   # turn on verbosity for all

    args.model_spec = model_spec
    logger.info("Parsed Arguments:" + str(args))  # Debug statement

    if args.detached:
        # send logging to zmq so that it will show in the parent log
        logger = get_logger(name="child", to_zmq=True, to_console=False)
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
        try:
            # Log request details for specific methods
            if request.method in ("POST", "PUT", "PATCH"):
                # Parse body and preserve it
                body_content, is_multipart = await parse_request_body(request)

                # Format content for logging
                if is_multipart:
                    request_content = "Multipart Form Data (File Upload)"
                elif isinstance(body_content, dict):
                    request_content = json.dumps(body_content, indent=2)
                elif isinstance(body_content, str):
                    try:
                        parsed_content = json.loads(body_content)
                        request_content = json.dumps(parsed_content, indent=2)
                    except json.JSONDecodeError:
                        request_content = body_content
                else:
                    request_content = str(body_content)

                logger.debug(
                    f"API Request:\n"
                    f"Method: {request.method}\n"
                    f"URL: {request.url}\n"
                    f"Headers: {dict(request.headers)}\n"
                    f"Content-Type: {request.headers.get('content-type', 'Not specified')}\n"
                    f"Body: {request_content if request_content else 'No Body'}"
                )

            # Proceed with the request
            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"Middleware error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error in middleware"}
            )

    if model_spec:
        # load model
        model_manager.load_model(model_spec)
        model_manager.model_ready = True

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)    # disable log_config to use our custom logger


def parse_dict(arg):
    """Parses a key=value string and returns a dictionary."""
    result = {}
    for pair in arg.split():
        key, value = pair.split('=')
        result[key] = value.strip("'")  # Strip single quotes here as well
    return result


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Launch local AI model")
    arg_parser.add_argument("--model-spec", type=str, help="Base64 encoded JSON ModelSpec object")
    arg_parser.add_argument('-v', "--verbose", action='store_true', help="Turn on more verbose logging")
    arg_parser.add_argument('-d', "--detached", action='store_true', help="Log to ZeroMQ")
    arg_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
    arg_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")
    arg_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    args = arg_parser.parse_args()

    make_server(args)
