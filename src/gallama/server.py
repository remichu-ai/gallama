import time
import sys
import httpx
import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from collections import defaultdict
from typing import Union
from gallama.data_classes.data_class import ModelParser, ModelObjectResponse, ModelObject, ModelDownloadSpec
from gallama.data_classes import ModelRequest, ModelInstanceInfo,  ModelInfo, AgentWithThinking, StopModelByPort
from gallama.server_engine import download_model_from_hf, handle_mixture_of_agent_request, create_options_response
from typing import List, Dict, Optional
from gallama.config import ConfigManager
from gallama.server_engine import forward_request
import psutil
import shutil
import asyncio
import uvicorn
import argparse
from gallama.logger.logger import get_logger
from gallama.utils.utils import get_package_file_path
from gallama.server_engine import log_model_status
import logging
from logging import DEBUG, INFO
import zmq
import os
import json
import threading
import traceback


manager_app = FastAPI()


@manager_app.middleware("http")
@manager_app.middleware("https")
async def log_requests(request: Request, call_next):
    try:
        if request.method in ("POST", "PUT", "PATCH"):  # Methods that typically have a body
            request_content = await request.body()
            if request_content:  # Only process if the body is not empty
                request_content = json.dumps(json.loads(request_content.decode("utf-8")), indent=2)
                logger.info(f"API Request:\nMethod: {request.method}\nURL: {request.url}\n{request_content}")

        response = await call_next(request)
    except RequestValidationError as e:
        logger.debug(f"Validation error:\n{e}")
        response = JSONResponse(status_code=422, content={"detail": "Validation error"})

    return response

# Add CORS middleware
manager_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


models: Dict[str, ModelInfo] = {}
model_load_queue = asyncio.Queue()
loading_lock = asyncio.Lock()
active_requests_lock = asyncio.Lock()
task_status = {}
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


# set up logging
def start_log_receiver(zmq_url):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    # Initialize the logger for the receiver
    receiver_logger = get_logger(name="log_receiver", to_console=True, to_file=False, to_zmq=False)

    receiver_logger.info(f"Log receiver started on {zmq_url}")

    def receive_logs():
        while True:
            try:
                message = socket.recv_json()
                log_level = getattr(logging, message['level'].upper(), logging.INFO)

                receiver_logger.log(
                    level=log_level,
                    msg=f"{message['model']}:{message['port']} | {message['log']}"
                )
            except zmq.Again:
                # No message available, sleep for a short time
                time.sleep(0.1)
            except zmq.ZMQError as e:
                receiver_logger.error(f"ZMQ Error in log receiver: {e}")
            except json.JSONDecodeError as e:
                receiver_logger.error(f"JSON Decode Error in log receiver: {e}")
            except Exception as e:
                receiver_logger.error(f"Unexpected error in log receiver: {e}")
                receiver_logger.error(traceback.format_exc())

    # Start the receiver in a separate thread
    receiver_thread = threading.Thread(target=receive_logs, daemon=True)
    receiver_thread.start()

    return receiver_thread



# Start the log receiver in a separate thread
DEFAULT_ZMQ_URL = "tcp://127.0.0.1:5555"  # Using 5559 as a standard port for logging


# Initialize the logger for the manager
# Set to_console=True and to_zmq=False to avoid duplication
logger = get_logger(name="manager", to_console=True, to_zmq=False)





@manager_app.post("/download_model")
def download_model(model_spec: ModelDownloadSpec):
    download_model_from_hf(model_spec)


async def model_loader():
    while True:
        logger.info("Model loader waiting for next model in queue")
        model = await model_load_queue.get()
        logger.info(f"Model loader retrieved model from queue: {model}")
        async with loading_lock:
            try:
                logger.info(f"Starting to load model: {model}")
                logger.info(f"Current queue size: {model_load_queue.qsize()}")
                await run_model_with_timeout(model)
            except Exception as e:
                logger.exception(f"Error loading model {model} instance: {str(e)}")
            finally:
                logger.info(f"Finished processing model: {model}")
                model_load_queue.task_done()
                logger.info(f"Remaining queue size: {model_load_queue.qsize()}")

        logger.info("Model loader finished processing, looping back")
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


async def run_model(model: ModelParser):
    try:
        # add the entry for the dictionary (where key is the model name)
        if model.model_name not in models:
            models[model.model_name] = ModelInfo(instances=[])

        # Find the next available port
        port = START_PORT
        while any(instance.port == port for model_info in models.values() for instance in model_info.instances):
            port += 1

        logger.info(f"Attempting to start model {model.model_name} on port {port}")

        model_config = config_manager.configs.get(model.model_name)
        backend = model.backend or model_config.get('backend')

        try:
            # Use the function
            app_path = get_package_file_path('app.py')
            logger.info(f"Using app path: {app_path}")

            # Determine the correct Python executable
            python_exec = shutil.which("python3") or shutil.which("python")

            if backend == "exllama":
                model_cli_args = model.to_arg_string()
                logger.debug(f"model cli: {model_cli_args}")
                process = await asyncio.create_subprocess_exec(
                    python_exec, app_path, "-id", model_cli_args, "--detached", "--port", str(port),
                    stdout=asyncio.subprocess.DEVNULL,
                    # stderr=asyncio.subprocess.DEVNULL,
                )
            else:
                # for embedding infinity doesnt have way to select GPU to be used, hence enforce
                # CUDA_VISIBLE_DEVICES constraints before launching fastapi

                # Create a copy of the current environment
                env = os.environ.copy()

                # Set CUDA_VISIBLE_DEVICES
                env['CUDA_VISIBLE_DEVICES'] = model.get_visible_gpu_indices()

                model_cli_args = model.to_arg_string()
                logger.debug(f"model cli: {model_cli_args}")
                process = await asyncio.create_subprocess_exec(
                    python_exec, app_path, "-id", model_cli_args, "--detached", "--port", str(port),
                    stdout=asyncio.subprocess.DEVNULL,
                    # stderr=asyncio.subprocess.DEVNULL,
                    env=env  # Pass the modified environment to the subprocess
                )

        except Exception as e:
            logger.error(f"Failed to create subprocess for model {model.model_id} on port {port}: {str(e)}")
            return

        # Wait for the model to become ready
        if await wait_for_model_ready(port):
            logger.info(f"Model {model.model_name} on port {port} is ready")
            instance_info = ModelInstanceInfo(
                port=port,
                pid=process.pid,
                status="running",
                model_id=model.model_id,
                embedding=model.backend == "embedding"
            )
            models[model.model_name].instances.append(instance_info)
        else:
            logger.error(f"Timeout waiting for model {model.model_name} on port {port} to become ready")
            await stop_model_instance(model.model_name, port)
            return

        logger.info(f"Model {model.model_id} instance on port {port} is fully loaded and ready")
        log_model_status(models, custom_logger=logger)  # Log status after successfully loading a model

        # Instead of entering an infinite loop, we'll exit the function here
        return

    except Exception as e:
        logger.exception(f"Error running model {model.model_name} instance on port {port}: {str(e)}")
        await stop_model_instance(model.model_name, port)
    finally:
        await cleanup_after_model_load(model.model_name)
        logger.info(f"Exiting run_model for {model.model_name} on port {port}")


async def cleanup_after_model_load(model: str):
    # Perform any necessary cleanup here
    logger.info(f"Performing cleanup after loading model: {model}")
    # For example, you might want to close any open connections or release resources
    # This is a placeholder - add specific cleanup tasks as needed


async def run_model_with_timeout(model: str, timeout: int = 600):  # 10 minutes timeout
    try:
        await asyncio.wait_for(run_model(model), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred while loading model: {model}")
        # Perform cleanup or error handling here
        await stop_model_instance(model, models[model].instances[-1].port if model in models and models[model].instances else None)


async def stop_model_instance(model: str, port: int):
    if model not in models:
        logger.warning(f"Attempted to stop instance of non-existent model: {model}")
        return

    instance = next((inst for inst in models[model].instances if inst.port == port), None)
    if not instance:
        logger.warning(f"Attempted to stop non-existent instance of model {model} on port {port}")
        return

    try:
        process = psutil.Process(instance.pid)
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            logger.warning(f"Model {model} instance on port {port} did not terminate gracefully. Forcing kill.")
            process.kill()

        if process.is_running():
            logger.error(f"Failed to kill process for model {model} instance on port {port}")
        else:
            logger.info(f"Successfully stopped model {model} instance on port {port}")
    except psutil.NoSuchProcess:
        logger.warning(f"Process for model {model} instance on port {port} no longer exists")
    except Exception as e:
        logger.exception(f"Error stopping model {model} instance on port {port}: {str(e)}")
    finally:
        models[model].instances = [inst for inst in models[model].instances if inst.port != port]
        if not models[model].instances:
            del models[model]
        logger.info(f"Cleaned up model {model} instance on port {port}")
        log_model_status(models, custom_logger=logger)   # Log status after removing a model instance


@manager_app.post("/v1/stop_model_by_port")
@manager_app.post("/stop_model_by_port")
async def stop_model_by_port(request: StopModelByPort):
    model_to_remove = None
    instance_to_remove = None

    # Find the model and instance by port
    for model, model_data in models.items():
        instance = next((inst for inst in model_data.instances if inst.port == request.port), None)
        if instance:
            model_to_remove = model
            instance_to_remove = instance
            break

    if not model_to_remove or not instance_to_remove:
        logger.warning(f"Attempted to stop non-existent instance on port {request.port}")
        return

    try:
        process = psutil.Process(instance_to_remove.pid)
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            logger.warning(f"Instance on port {request.port} did not terminate gracefully. Forcing kill.")
            process.kill()

        if process.is_running():
            logger.error(f"Failed to kill process for instance on port {request.port}")
        else:
            logger.info(f"Successfully stopped instance on port {request.port}")
    except psutil.NoSuchProcess:
        logger.warning(f"Process for instance on port {request.port} no longer exists")
    except Exception as e:
        logger.exception(f"Error stopping instance on port {request.port}: {str(e)}")
    finally:
        # Remove the instance from the model
        models[model_to_remove].instances = [
            inst for inst in models[model_to_remove].instances if inst.port != request.port
        ]

        # If no instances remain, remove the model entirely
        if not models[model_to_remove].instances:
            del models[model_to_remove]

        logger.info(f"Cleaned up instance on port {request.port}")
        log_model_status(models, custom_logger=logger)  # Log status after removal


async def get_model_from_body(request: Request) -> str:
    try:
        body = await request.json()
        return body.get("model", "")
    except json.JSONDecodeError:
        return ""


# Add a health check endpoint
@manager_app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Add a periodic health check function
async def periodic_health_check():
    while True:
        await asyncio.sleep(60)  # Check every minute
        for model_name, model_info in models.items():
            for instance in model_info.instances:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"http://localhost:{instance.port}/health", timeout=5.0)
                        if response.status_code != 200:
                            logger.warning(f"Instance {instance.port} of model {model_name} is not healthy")
                            # Implement recovery logic here (e.g., restart the instance)
                except Exception as e:
                    logger.error(f"Error checking health of instance {instance.port} of model {model_name}: {str(e)}")
                    # Implement recovery logic here (e.g., restart the instance)


# Proposed fix:
async def load_models(model_request, task_id):
    task_status[task_id] = "loading"
    try:
        models_to_load = model_request if isinstance(model_request, List) else [model_request]
        for model in models_to_load:
            await model_load_queue.put(model)
            logger.info(f"Model {model} instance queued for loading")

        # Wait for all models to be loaded
        await model_load_queue.join()

        task_status[task_id] = "completed"
    except Exception as e:
        task_status[task_id] = f"failed: {str(e)}"

@manager_app.post("/add_model")
@manager_app.post("/v1/add_model")
async def add_model(model_request: Union[ModelParser, List[ModelParser]], background_tasks: BackgroundTasks, request: Request):
    task_id = str(uuid.uuid4())
    # Log the incoming request body
    logger.info(f"Received raw request: {await request.body()}")


    logger.info(f"Received request to add model(s): {model_request}, Task ID: {task_id}")

    if any(instance.status == "loading" for model in models.values() for instance in model.instances):
        logger.warning("Another model is currently loading. Request rejected.")
        raise HTTPException(status_code=409, detail="Another model is currently loading. Please try again later.")

    task_status[task_id] = "queued"
    background_tasks.add_task(load_models, model_request, task_id)

    return {"message": f"Model(s) queued for loading", "task_id": task_id}


@manager_app.get("/task_status/{task_id}")
@manager_app.get("/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    status = task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": status}


@manager_app.post("/remove_model")
@manager_app.post("/v1/remove_model")
async def remove_model(model_request: ModelRequest):
    if model_request.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    for instance in models[model_request.model_id].instances:
        await stop_model_instance(model_request.model_id, instance.port)
    return {"message": f"All instances of model {model_request.model_id} removal initiated"}



@manager_app.get("/v1/list_loaded_models")
@manager_app.get("/list_loaded_models")
async def list_models():
    return {name: {"instances": [{"port": inst.port, "status": inst.status} for inst in model.instances]}
            for name, model in models.items()}


@manager_app.get("/list_available_models")
@manager_app.get("/v1/list_available_models")
async def list_available_models():
    config_manager = ConfigManager()
    downloaded_models = config_manager.list_downloaded_models_dict
    # logger.info(downloaded_models)
    return downloaded_models


@manager_app.get("/v1/models")
async def get_models(request: Request):
    data = []
    for name, model in models.items():
        data.append(ModelObject(id=name))

    return ModelObjectResponse(data=data)


@manager_app.get("/loading_status")
async def loading_status():
    return {
        "queue_size": model_load_queue.qsize(),
        "currently_loading": loading_lock.locked(),
        "models": {name: {"instances": [{"port": inst.port, "status": inst.status} for inst in model.instances]}
                   for name, model in models.items()}
    }


async def get_instance_for_model(model: str):
    running_instances = [inst for inst in models[model].instances if inst.status == "running"]
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
    body = await request.body()
    body_json = json.loads(body)

    if request.url.path in EXCLUDED_ENDPOINTS:
        logger.info(body_json)
        return await request.app.router.get_route_handler()(request)

    if request.method == "OPTIONS":
        return create_options_response(dict(request.headers))

    if body_json.get("mixture_of_agents", False):
        return await handle_mixture_of_agent_request(request, body_json, models, active_requests)

    model = await get_model_from_body(request)

    async with request_semaphore:
        is_embedding = any(subpath["original"] in path for subpath in EMBEDDING_SUBPATHS)

        if not model and strict_mode:
            raise HTTPException(status_code=400, detail="Model must be specified in strict mode")

        available_instances = []

        if strict_mode:
            if model not in models:
                raise HTTPException(status_code=404, detail="Specified model not found")
            available_instances = [inst for inst in models[model].instances if inst.status == "running"]
        else:
            # Try to find a matching model first
            if model:
                if model in models:
                    available_instances = [inst for inst in models[model].instances if inst.status == "running"]

            # If no matching model or no instances found, pick any running instance
            if not available_instances:
                for model_info in models.values():
                    for inst in model_info.instances:
                        if inst.status == "running":
                            if is_embedding:
                                # For embedding requests, select instances with matching model name
                                if not model or inst.model_id == model:
                                    available_instances.append(inst)
                            else:
                                # For non-embedding requests, select all non-embedding instances
                                if not inst.embedding:
                                    available_instances.append(inst)

        if not available_instances:
            raise HTTPException(status_code=503, detail=f"No suitable running instances with requested model '{model}'")

        # Select the instance with the least active requests
        instance = min(available_instances, key=lambda inst: active_requests[inst.port])

        # Increment the active request count for the selected instance
        async with active_requests_lock:
            active_requests[instance.port] += 1

        try:
            logger.info(f"active_requests: {str(active_requests)}")
            logger.info(f"Request routed to model {instance.model_id} instance at port {instance.port}")

            # Forward the request
            response = await forward_request(request, instance)

            return response
        finally:
            # Decrement the active request count after the request is completed
            async with active_requests_lock:
                active_requests[instance.port] -= 1


@manager_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def router(request: Request, path: str):
    return await load_balanced_router(request, path)


def shutdown(signal, frame):
    logger.info("Shutting down all models...")
    for model in list(models.keys()):
        for instance in models[model].instances:
            stop_model_instance(model, instance.port)
    logger.info("All models have been shut down. Exiting.")
    sys.exit(0)


async def start_server(port=8000):
    config = uvicorn.Config(manager_app, host="0.0.0.0", port=port, log_level="debug")
    server = uvicorn.Server(config)
    await server.serve()


async def main(model_list=None, port=8000, strict_mode=False):
    receiver_thread = start_log_receiver(DEFAULT_ZMQ_URL)
    logger.info("Starting main function")
    logger.info(f"Strict mode: {'enabled' if strict_mode else 'disabled'}")

    model_loader_task = asyncio.create_task(model_loader())
    logger.info("Created model_loader task")

    # Load initial models
    if model_list:
        logger.info(f"Loading initial models: {[model.model_id for model in model_list]}")
        for model in model_list:
            await model_load_queue.put(model)

        # Wait for all initial models to load
        while not model_load_queue.empty():
            await asyncio.sleep(1)

        logger.info("All initial models have been queued for loading")

    # Start periodic health checks
    health_check_task = asyncio.create_task(periodic_health_check())

    await start_server(port)

    # Ensure tasks are cancelled when the server_engine stops
    model_loader_task.cancel()
    health_check_task.cancel()
    try:
        await asyncio.gather(model_loader_task, health_check_task)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled")


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
    logger.info(llama_art)


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
            model_spec = ModelParser.from_dict(item)
            model_list.append(model_spec)
            print(model_spec)

    if model_list:
        logger.info("Initial models: " + str(model_list))

    # set logger level
    if args.verbose:
        logger.setLevel(DEBUG)
    else:
        logger.setLevel(INFO)


    logger.info("Parsed Arguments:" + str(args))

    asyncio.run(
        main(
            model_list=model_list,
            port=args.port,
            strict_mode=args.strict_mode
        )
    )


if __name__ == "__main__":

    logger.info("Script started")
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

