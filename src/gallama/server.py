import time
import sys
import httpx
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from collections import defaultdict
from typing import Union
from gallama.data_classes.data_class import ModelSpec
from gallama.data_classes import  ModelInstanceInfo,  ModelInfo, AgentWithThinking
from gallama.server_engine import handle_mixture_of_agent_request, create_options_response
from gallama.utils import parse_request_body
from typing import List, Dict
from gallama.config import ConfigManager
from gallama.server_engine import forward_request
import shutil
import asyncio
import uvicorn
import argparse
from gallama.utils.utils import get_package_file_path
from logging import DEBUG, INFO
import os
import json
import base64
from gallama.server_routes import (
    server_management_router,
    realtime_router,
    periodic_health_check,
    stop_model_instance
)
from gallama.server_engine import log_model_status
from gallama.dependencies_server import get_server_manager, get_server_logger, start_log_receiver, DEFAULT_ZMQ_URL


server_logger = get_server_logger()

router = APIRouter()
router.include_router(server_management_router)
router.include_router(realtime_router)


manager_app = FastAPI()
manager_app.include_router(router)





@manager_app.middleware("http")
@manager_app.middleware("https")
async def log_requests(request: Request, call_next):
    try:
        if request.method in ("POST", "PUT", "PATCH"):  # Methods that typically have a body
            content_type = request.headers.get("Content-Type", "")
            if "multipart/form-data" in content_type or "application/octet-stream" in content_type:
                server_logger.info(f"API Request:\nMethod: {request.method}\nURL: {request.url}\n[Binary content omitted]")
            else:
                request_content = await request.body()
                if request_content:
                    try:
                        request_content = json.dumps(json.loads(request_content.decode("utf-8")), indent=2)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        request_content = "[Non-JSON or binary content omitted]"
                    server_logger.info(f"API Request:\nMethod: {request.method}\nURL: {request.url}\n{request_content}")

        response = await call_next(request)
    except RequestValidationError as e:
        server_logger.debug(f"Validation error:\n{e}")
        response = JSONResponse(status_code=422, content={"detail": "Validation error"})

    return response


class WebSocketLoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            await self.log_websocket(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def log_websocket(self, scope: Scope, receive: Receive, send: Send):
        # Log detailed connection attempt information
        client = scope.get('client')
        server_logger.info(f"WebSocket connection attempt from {client}")

        # Log query parameters
        query_string = scope.get('query_string', b'').decode('utf-8')
        server_logger.debug(f"Query parameters: {query_string}")

        # Convert headers to a more readable format and log them
        headers = dict(scope.get('headers', []))
        formatted_headers = {
            k.decode('utf-8'): v.decode('utf-8')
            for k, v in headers.items()
        }
        server_logger.debug("WebSocket headers:")
        for key, value in formatted_headers.items():
            server_logger.debug(f"  {key}: {value}")

        # Log protocols specifically
        protocols = formatted_headers.get('sec-websocket-protocol', '').split(', ')
        server_logger.debug(f"Requested protocols: {protocols}")

        async def receive_with_logging():
            message = await receive()
            server_logger.debug(f"Received message type: {message['type']}")

            if message["type"] == "websocket.connect":
                server_logger.info("WebSocket client connecting")
                # Log any authentication info (safely)
                auth_header = formatted_headers.get('authorization', '')
                if auth_header:
                    server_logger.debug("Authorization header present")

            elif message["type"] == "websocket.disconnect":
                server_logger.info(f"WebSocket client disconnecting from {client}")

            elif message["type"] == "websocket.receive":
                try:
                    if "text" in message:
                        content = message["text"]
                        try:
                            parsed_content = json.loads(content)
                            server_logger.info(
                                "WebSocket received text:\n" +
                                json.dumps(parsed_content, indent=2)
                            )
                        except json.JSONDecodeError:
                            server_logger.info(f"WebSocket received raw text: {content}")
                    elif "bytes" in message:
                        server_logger.info(f"WebSocket received binary data of size: {len(message['bytes'])} bytes")
                except Exception as e:
                    server_logger.error(f"Error logging WebSocket message: {str(e)}")
                    server_logger.debug(f"Raw message content: {message}")
            return message

        async def send_with_logging(message):
            server_logger.debug(f"Sending message type: {message['type']}")

            if message["type"] == "websocket.accept":
                selected_protocol = message.get('subprotocol')
                server_logger.info(f"WebSocket connection accepted with protocol: {selected_protocol}")

            elif message["type"] == "websocket.close":
                code = message.get('code', 1000)
                reason = message.get('reason', '')
                server_logger.info(f"WebSocket connection closed with code {code}: {reason}")

            elif message["type"] == "websocket.send":
                try:
                    if "text" in message:
                        content = message["text"]
                        try:
                            parsed_content = json.loads(content)
                            server_logger.info(
                                "WebSocket sent text:\n" +
                                json.dumps(parsed_content, indent=2)
                            )
                        except json.JSONDecodeError:
                            server_logger.info(f"WebSocket sent raw text: {content}")
                    elif "bytes" in message:
                        server_logger.info(f"WebSocket sent binary data of size: {len(message['bytes'])} bytes")
                except Exception as e:
                    server_logger.error(f"Error logging WebSocket message: {str(e)}")
                    server_logger.debug(f"Raw message content: {message}")

            # Log any error responses
            if message.get('status', 200) >= 400:
                server_logger.error(f"WebSocket error response: {message}")

            await send(message)

        try:
            await self.app(scope, receive_with_logging, send_with_logging)
        except Exception as e:
            server_logger.error(f"WebSocket connection error: {str(e)}")
            server_logger.debug(f"Error details:", exc_info=True)

# Add WebSocket middleware
# manager_app.add_middleware(WebSocketLoggingMiddleware)

# Add CORS middleware
manager_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# models: Dict[str, ModelInfo] = {}
# model_load_queue = asyncio.Queue()
# loading_lock = asyncio.Lock()
# active_requests_lock = asyncio.Lock()
# task_status = {}
config_manager = ConfigManager()

# Define a variable for the starting port
START_PORT = 8001

# Increase the timeout for long-running tasks (adjust as needed)
TIMEOUT = 300  # 5 minutes

strict_mode = False

# List of endpoints to exclude from API gateway redirection
EXCLUDED_ENDPOINTS = [
    "/add_model", "/remove_model", "/list_models", "/list_available_models",
    "/v1/add_model", "/v1/remove_model", "/v1/list_models", "/v1/list_available_models",
    "/list_models", "/v1/list_models"
    "/v1/remove_model_by_port","/remove_model_by_port",
    "/v1/models", "/task_status"
]
EMBEDDING_SUBPATHS = []     # No overwriting at the moment
# EMBEDDING_SUBPATHS = [
#     {
#         "original": "/v1/embeddings",
#         "replacement": "/v1/embeddings"
#     },
#     # Add more paths here as needed, e.g.:
#     # {"original": "/v2/some_path", "replacement": "/new_path"}
# ]

# Dictionary to keep track of active requests per instance
active_requests: Dict[int, int] = defaultdict(int)

# Add a semaphore for concurrency control
MAX_CONCURRENT_REQUESTS = 100  # Adjust this value based on your system's capacity
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


# # set up logging
# def start_log_receiver(zmq_url):
#     context = zmq.Context()
#     socket = context.socket(zmq.PULL)
#     socket.bind("tcp://*:5555")
#
#     # Initialize the server_logger for the receiver
#     receiver_logger = get_logger(name="log_receiver", to_console=True, to_file=False, to_zmq=False)
#
#     receiver_logger.info(f"Log receiver started on {zmq_url}")
#
#     def receive_logs():
#         while True:
#             try:
#                 message = socket.recv_json()
#                 log_level = getattr(logging, message['level'].upper(), logging.INFO)
#
#                 receiver_logger.log(
#                     level=log_level,
#                     msg=f"{message['model']}:{message['port']} | {message['log']}"
#                 )
#             except zmq.Again:
#                 # No message available, sleep for a short time
#                 time.sleep(0.1)
#             except zmq.ZMQError as e:
#                 receiver_logger.error(f"ZMQ Error in log receiver: {e}")
#             except json.JSONDecodeError as e:
#                 receiver_logger.error(f"JSON Decode Error in log receiver: {e}")
#             except Exception as e:
#                 receiver_logger.error(f"Unexpected error in log receiver: {e}")
#                 receiver_logger.error(traceback.format_exc())
#
#     # Start the receiver in a separate thread
#     receiver_thread = threading.Thread(target=receive_logs, daemon=True)
#     receiver_thread.start()
#
#     return receiver_thread
#
#
#
# # Start the log receiver in a separate thread
# DEFAULT_ZMQ_URL = "tcp://127.0.0.1:5555"  # Using 5559 as a standard port for logging
#
#
# # Initialize the logger for the manager
# # Set to_console=True and to_zmq=False to avoid duplication
# logger = get_logger(name="manager", to_console=True, to_zmq=False)



async def model_loader():
    server_manager = get_server_manager()

    while True:
        server_logger.info("Model loader waiting for next model in queue")
        model = await server_manager.model_load_queue.get()
        server_logger.info(f"Model loader retrieved model from queue: {model}")
        async with server_manager.loading_lock:
            try:
                server_logger.info(f"Starting to load model: {model}")
                server_logger.info(f"Current queue size: {server_manager.model_load_queue.qsize()}")
                await run_model_with_timeout(model)
            except Exception as e:
                server_logger.exception(f"Error loading model {model} instance: {str(e)}")
            finally:
                server_logger.info(f"Finished processing model: {model}")
                server_manager.model_load_queue.task_done()
                server_logger.info(f"Remaining queue size: {server_manager.model_load_queue.qsize()}")

        server_logger.info("Model loader finished processing, looping back")
        await asyncio.sleep(1)


async def wait_for_model_ready(port, timeout=300):  # 5 minutes timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/health", timeout=5.0)
                if response.status_code == 200:
                    return True
        except httpx.RequestError:
            pass  # The server_engine might not be up yet, so we'll just continue waiting
        await asyncio.sleep(1)
    return False


async def run_model(model: ModelSpec):
    server_manager = get_server_manager()

    try:
        # add the entry for the dictionary (where key is the model name)
        if model.model_name not in server_manager.models:
            server_manager.models[model.model_name] = ModelInfo(instances=[])

        # Find the next available port
        port = START_PORT
        while any(instance.port == port for model_info in server_manager.models.values() for instance in model_info.instances):
            port += 1

        server_logger.info(f"Attempting to start model {model.model_name} on port {port}")

        model_config = config_manager.configs.get(model.model_name)
        backend = model.backend or model_config.get('backend')

        try:
            # Serialize the ModelSpec to JSON and encode to base64
            model_json = model.model_dump_json()
            model_b64 = base64.b64encode(model_json.encode('utf-8')).decode('utf-8')


            # Use the function
            app_path = get_package_file_path('app.py')
            server_logger.info(f"Using app path: {app_path}")

            # Determine the correct Python executable
            python_exec = shutil.which("python3") or shutil.which("python")

            if backend == "exllama":
                # model_cli_args = model.to_arg_string()
                # server_logger.debug(f"model cli: {model_cli_args}")
                process = await asyncio.create_subprocess_exec(
                    python_exec, app_path, "--model-spec", model_b64, "--detached", "--port", str(port),
                    # stdout=asyncio.subprocess.DEVNULL,
                    # stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                # other than exllama, we will use env setting to set visible GPUs
                # CUDA_VISIBLE_DEVICES constraints before launching fastapi

                # Create a copy of the current environment
                env = os.environ.copy()

                # Set CUDA_VISIBLE_DEVICES
                env['CUDA_VISIBLE_DEVICES'] = model.get_visible_gpu_indices()

                # model_cli_args = model.to_arg_string()
                # server_logger.debug(f"model cli: {model_cli_args}")
                process = await asyncio.create_subprocess_exec(
                    python_exec, app_path, "--model-spec", model_b64, "--detached", "--port", str(port),
                    stdout=asyncio.subprocess.DEVNULL,
                    # stderr=asyncio.subprocess.DEVNULL,
                    env=env  # Pass the modified environment to the subprocess
                )

        except Exception as e:
            server_logger.error(f"Failed to create subprocess for model {model.model_id} on port {port}: {str(e)}")
            return

        # Wait for the model to become ready
        if await wait_for_model_ready(port):
            server_logger.info(f"Model {model.model_name} on port {port} is ready")
            instance_info = ModelInstanceInfo(
                port=port,
                pid=process.pid,
                status="running",
                model_name=model.model_name,
                model_type=ModelSpec.get_model_type_from_backend(backend)
            )
            server_manager.models[model.model_name].instances.append(instance_info)
        else:
            server_logger.error(f"Timeout waiting for model {model.model_name} on port {port} to become ready")
            await stop_model_instance(model.model_name, port)
            return

        server_logger.info(f"Model {model.model_id} instance on port {port} is fully loaded and ready")
        log_model_status(server_manager.models, custom_logger=server_logger)  # Log status after successfully loading a model

        # Instead of entering an infinite loop, we'll exit the function here
        return

    except Exception as e:
        server_logger.exception(f"Error running model {model.model_name} instance on port {port}: {str(e)}")
        await stop_model_instance(model.model_name, port)
    finally:
        await cleanup_after_model_load(model.model_name)
        server_logger.info(f"Exiting run_model for {model.model_name} on port {port}")


async def cleanup_after_model_load(model: str):
    # Perform any necessary cleanup here
    server_logger.info(f"Performing cleanup after loading model: {model}")
    # For example, you might want to close any open connections or release resources
    # This is a placeholder - add specific cleanup tasks as needed


async def run_model_with_timeout(model: str, timeout: int = 600):  # 10 minutes timeout
    server_manager = get_server_manager()

    try:
        await asyncio.wait_for(run_model(model), timeout=timeout)
    except asyncio.TimeoutError:
        server_logger.error(f"Timeout occurred while loading model: {model}")
        # Perform cleanup or error handling here
        await stop_model_instance(model, server_manager.models[model].instances[-1].port if model in server_manager.models and server_manager.models[model].instances else None)




async def get_model_from_body(request: Request) -> str:
    try:
        body, is_multipart = await parse_request_body(request)
        return body.get("model", "")
    except Exception as e:     # when request doesnt come with model e.g. audio transcription
        server_logger.debug(f"Can not get body from request {e}. Hence continue with raw body")
        return ""



async def get_instance_for_model(model: str):
    server_manager = get_server_manager()

    running_instances = [inst for inst in server_manager.models[model].instances if inst.status == "running"]
    if not running_instances:
        raise HTTPException(status_code=503, detail=f"No running instances available for model: {model}")

    return min(running_instances, key=lambda inst: active_requests[inst.port])


async def forward_to_multiple_agents(request: Request, agent_list: List[Union[str, AgentWithThinking]], modified_body: str, modified_headers: str):
    tasks = []
    for agent in agent_list:
        if isinstance(agent, str):
            instance = await get_instance_for_model(agent)
            tasks.append(forward_request(request, instance, modified_body, modified_headers))
        elif isinstance(agent, AgentWithThinking):
            instance = await get_instance_for_model(agent.model)
            if agent.thinking_template:
                # Modify the request to include the thinking in the thinking_template field
                agent_body = json.loads(modified_body)
                agent_body["thinking_template"] = agent.thinking_template
                agent_modified_body = json.dumps(agent_body).encode()
                agent_modified_headers = dict(modified_headers)
                agent_modified_headers["content-length"] = str(len(agent_modified_body))
                tasks.append(forward_request(request, instance, agent_modified_body, agent_modified_headers))
            else:
                tasks.append(forward_request(request, instance, modified_body, modified_headers))
    return await asyncio.gather(*tasks)



async def load_balanced_router(request: Request, path: str):
    server_manager = get_server_manager()

    body_json, is_multipart = await parse_request_body(request)

    if request.url.path in EXCLUDED_ENDPOINTS:
        server_logger.info(body_json)
        return await request.app.router.get_route_handler()(request)

    if request.method == "OPTIONS":
        return create_options_response(dict(request.headers))

    if isinstance(body_json, dict) and body_json.get("mixture_of_agents", False):
        return await handle_mixture_of_agent_request(request, body_json, server_manager.models, active_requests)

    model = await get_model_from_body(request)

    async with request_semaphore:
        is_embedding = any(subpath["original"] in path for subpath in EMBEDDING_SUBPATHS)

        if not model and strict_mode:
            raise HTTPException(status_code=400, detail="Model must be specified in strict mode")

        available_instances = []

        if strict_mode:
            if model not in server_manager.models:
                raise HTTPException(status_code=404, detail="Specified model not found")
            available_instances = [inst for inst in server_manager.models[model].instances if inst.status == "running"]
        else:
            # Try to find a matching model first
            if model:
                if model in server_manager.models:
                    available_instances = [inst for inst in server_manager.models[model].instances if inst.status == "running"]

            # If no matching model or no instances found, pick any running instance
            if not available_instances:
                for model_info in server_manager.models.values():
                    for inst in model_info.instances:
                        if inst.status == "running":
                            if is_embedding:
                                # For embedding requests, select instances with matching model name
                                if not model or inst.model_name == model:
                                    available_instances.append(inst)
                            else:
                                # For non-embedding requests, select all non-embedding instances
                                if inst.model_type != "embedding":
                                    available_instances.append(inst)

        if not available_instances:
            raise HTTPException(status_code=503, detail=f"No suitable running instances with requested model '{model}'")

        # Select the instance with the least active requests
        instance = min(available_instances, key=lambda inst: active_requests[inst.port])

        # Increment the active request count for the selected instance
        async with server_manager.active_requests_lock:
            active_requests[instance.port] += 1

        try:
            server_logger.info(f"active_requests: {str(active_requests)}")
            server_logger.info(f"Request routed to model {instance.model_name} instance at port {instance.port}")

            # Forward the request
            response = await forward_request(request, instance)

            return response
        finally:
            # Decrement the active request count after the request is completed
            async with server_manager.active_requests_lock:
                active_requests[instance.port] -= 1


@manager_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def router(request: Request, path: str):
    return await load_balanced_router(request, path)


def shutdown(signal, frame):
    server_manager = get_server_manager()

    server_logger.info("Shutting down all models...")
    for model in list(server_manager.models.keys()):
        for instance in server_manager.models[model].instances:
            stop_model_instance(model, instance.port)
    server_logger.info("All models have been shut down. Exiting.")
    sys.exit(0)


async def start_server(port=8000):
    config = uvicorn.Config(manager_app, host="0.0.0.0", port=port, log_level="debug")
    server = uvicorn.Server(config)
    await server.serve()


async def main(model_list=None, port=8000, strict_mode=False):
    server_manager = get_server_manager()

    receiver_thread = start_log_receiver(DEFAULT_ZMQ_URL)
    server_logger.info("Starting main function")
    server_logger.info(f"Strict mode: {'enabled' if strict_mode else 'disabled'}")

    model_loader_task = asyncio.create_task(model_loader())
    server_logger.info("Created model_loader task")

    # Load initial models
    if model_list:
        server_logger.info(f"Loading initial models: {[model.model_name for model in model_list]}")
        for model in model_list:
            await server_manager.model_load_queue.put(model)

        # Wait for all initial models to load
        while not server_manager.model_load_queue.empty():
            await asyncio.sleep(1)

        server_logger.info("All initial models have been queued for loading")

    # Start periodic health checks
    health_check_task = asyncio.create_task(periodic_health_check())

    await start_server(port)

    # Ensure tasks are cancelled when the server_engine stops
    model_loader_task.cancel()
    health_check_task.cancel()
    try:
        await asyncio.gather(model_loader_task, health_check_task)
    except asyncio.CancelledError:
        server_logger.info("Tasks cancelled")


# this parser is different to app.py
def parse_dict(arg):
    """Parses a key=value string and returns a dictionary."""
    result = {}
    for pair in arg.split():
        key, value = pair.split('=')
        result[key] = value.strip("'")  # Strip single quotes here as well
    return result


def llama_picture():
    """ show a fun picture cause why not :) """
    # Load the image
    llama_art = """
        ,,__
        .. \\(oo)
           (__)
             ||----w |
             ||     ||
    """
    server_logger.info(llama_art)


def run_from_script(args):
    # opening llama picture cause why not
    llama_picture()

    global strict_mode

    if args.strict_mode:
        strict_mode = True

    # parse the cli input
    model_list = []
    if args.model_id:
        for item in args.model_id:
            model_spec = ModelSpec.from_dict(item)
            model_list.append(model_spec)
            server_logger.info(model_spec)

    # simple loading with model_name
    if args.model_name:
        model_spec = ModelSpec(model_name=args.model_name)
        model_list.append(model_spec)

    if model_list:
        server_logger.info("Initial models: " + str(model_list))

    # set server_logger level
    if args.verbose:
        server_logger.setLevel(DEBUG)
    else:
        server_logger.setLevel(INFO)


    server_logger.info("Parsed Arguments:" + str(args))

    asyncio.run(
        main(
            model_list=model_list,
            port=args.port,
            strict_mode=args.strict_mode
        )
    )


if __name__ == "__main__":

    server_logger.info("Script started")
    arg_parser = argparse.ArgumentParser(description="Launch multi model src instance")
    arg_parser.add_argument("--strict_mode", action="store_true", default=False,
                            help="Enable strict mode for routing non-embedding requests to matching model names")
    arg_parser.add_argument("-id", "--model_id", action='append', type=parse_dict, default=None,
                            help="Model configuration. Can be used multiple times for different models. "
                                 "Format: -id model_id=<model_name> [gpu=<vram1>,<vram2>,...] [gpu<n>=<vram>] [cache=<size>] [model_type=exllama|embedding]"
                                 "Examples:\n"
                                 "1. Specify all GPUs: -id model_id=/path/llama3-8b gpu=4,2,0,3 cache=8192\n"
                                 "2. Specify individual GPUs: -id model_id=/path/llama3-8b gpu0=8 gpu1=8 cache=8192\n"
                                 "3. Mix formats: -id model_id=gpt4 gpu=4,2 gpu3=3 cache=8192\n"
                                 "4. Embedding model: -id model_id=/path/gte-large-en-v1.5 gpu=0,0,0,3"
                                 "cache_size is defaulted to model context length if not specified"
                                 "cache_size can be more than context length, which will help model perform better in batched generation"
                                 "cache_size is not application to embedding model"
                                 "VRAM is specified in GB. Cache size is integer which is the context length to cache."
                                 "VRAM for embedding will simple set env parameter to allow infinity_embedding to view the specific GPU and can not enforce VRAM size restriction")
    arg_parser.add_argument('-v', "--verbose", action='store_true', help="Turn on more verbose logging")
    arg_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
    arg_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")

    # arg_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    args = arg_parser.parse_args()

    run_from_script(args)

