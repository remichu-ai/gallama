import subprocess
import textwrap
import time
import sys
import httpx
from fastapi import FastAPI, HTTPException, Request
from collections import defaultdict
from typing import Union, AsyncGenerator
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from gallama.data_class import ModelParser, ModelObjectResponse, ModelObject
from typing import List, Dict, Optional
import psutil
import asyncio
import uvicorn
import argparse
from .logger import get_logger
from .utils import get_package_file_path
import logging
from logging import DEBUG, INFO
import zmq
import os
import json
import threading
import traceback
from gallama.data_class import ChatCompletionResponse, BaseMessage
from copy import deepcopy

# Try to import torch, but don't raise an error if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelRequest(BaseModel):
    model_id: str

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInstanceInfo(BaseModel):
    model_id: str
    port: int
    pid: int  # Changed from process to pid
    status: str
    embedding: bool = False

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInfo(BaseModel):
    instances: List[ModelInstanceInfo]


class AgentWithThinking(BaseModel):
    model: str = Field(description="The name of the model")
    thinking_template: Optional[str] = Field(default=None, description="The XML thinking for the agent")


class MixtureOfAgents(BaseModel):
    agent_list: List[Union[str, AgentWithThinking]]


manager_app = FastAPI()
models: Dict[str, ModelInfo] = {}
model_load_queue = asyncio.Queue()
loading_lock = asyncio.Lock()
active_requests_lock = asyncio.Lock()

# Define a variable for the starting port
START_PORT = 8001

# Increase the timeout for long-running tasks (adjust as needed)
TIMEOUT = 300  # 5 minutes

strict_mode = False

# List of endpoints to exclude from API gateway redirection
EXCLUDED_ENDPOINTS = ["/add_model", "/remove_model", "/list_models", "/v1/models"]
EMBEDDING_SUBPATHS = [
    {
        "original": "/v1/embeddings",
        "replacement": "/v1/embeddings"
    },
    {
        "original": "v1/embeddings",
        "replacement": "v1/embeddings"
    }
    # Add more paths here as needed, e.g.:
    # {"original": "/v2/some_path", "replacement": "/new_path"}
]

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



# Add CORS middleware
manager_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_gpu_memory_info():
    if not TORCH_AVAILABLE:
        return "PyTorch not available, GPU info cannot be retrieved"

    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        lines = result.strip().split('\n')

        total_used = 0
        total_free = 0
        total_memory = 0

        gpu_memory_info = []
        for line in lines:
            index, used, free, total = map(int, line.split(','))
            gpu_memory_info.append(
                f"GPU {index}: "
                f"Used:  {used/1024:.1f}GB, "
                f"Free:  {free/1024:.1f}GB, "
                f"Total:  {total/1024:.1f}GB"
            )
            total_used += used
            total_free += free
            total_memory += total

        total_line = (f"Total: "
                      f"Used: {total_used/1024:.1f}GB, "
                      f"Free: {total_free/1024:.1f}GB, "
                      f"Total: {total_memory/1024:.1f}GB")

        if len(gpu_memory_info) == 1:
            return "\n".join(gpu_memory_info)   # no total as only 1 GPU
        else:
            return "\n".join(gpu_memory_info +
                             ["------+-------------+-------------+------------------------------------------+"] +
                             [total_line])    # add total line as more than 1 GPU
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unable to retrieve GPU information"


def log_model_status():
    total_models = len(models)
    total_instances = sum(len(model_info.instances) for model_info in models.values())

    # Prepare model details
    model_details = []
    for model_name, model_info in models.items():
        instances = ["{0}".format(inst.port) for inst in model_info.instances]
        model_details.append("| {0:<20} | {1:>2} | {2:<30} ".format(model_name, len(instances), ', '.join(instances)))

    # Prepare GPU info
    gpu_info = get_gpu_memory_info().split('\n')
    formatted_gpu_info = ''.join("| {0}\n".format(line) for line in gpu_info)

    # Construct the log message
    log_message = """```
+------------------------------------------------------------------------------+
| Current Status: {0} model(s) loaded with {1} total instance(s)               
+------------------------------------------------------------------------------+
| Model Name           | # | Ports                                             
+----------------------+---+---------------------------------------------------+
{2}
+------------------------------------------------------------------------------+
| GPU Memory Information                                                       
+-------+-------------+-------------+------------------------------------------+
| GPU   | Used        | Free        | Total       
+-------+-------------+-------------+------------------------------------------+
{3}+-------+-------------+-------------+------------------------------------------+
```""".format(
        total_models,
        total_instances,
        '\n'.join(model_details),
        formatted_gpu_info
    )

    logger.info(log_message)


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
            pass  # The server might not be up yet, so we'll just continue waiting
        await asyncio.sleep(1)
    return False


def get_library_path():
    # Get the directory containing the current file
    current_file_path = os.path.abspath(__file__)
    # Get the parent directory (which should be the library root)
    library_path = os.path.dirname(os.path.dirname(current_file_path))
    return library_path


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

        try:
            # Use the function
            app_path = get_package_file_path('app.py')
            logger.info(f"Using app path: {app_path}")
            # Create a copy of the current environment
            env = os.environ.copy()

            # Set CUDA_VISIBLE_DEVICES
            # this is because infinity embedding do not have gpu arguments
            # hence set it via env parameter
            env['CUDA_VISIBLE_DEVICES'] = model.get_visible_gpu_indices()

            model_cli_args = model.to_arg_string()
            logger.debug(f"model cli: {model_cli_args}")
            process = await asyncio.create_subprocess_exec(
                "python", app_path, "-id", model_cli_args, "--detached", "--port", str(port),
                stdout=asyncio.subprocess.DEVNULL,
                # stderr=asyncio.subprocess.DEVNULL,
                #env=env  # Pass the modified environment to the subprocess
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
        log_model_status()  # Log status after successfully loading a model

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
        log_model_status()  # Log status after removing a model instance


# Optimize the forward_request function
async def forward_request(request: Request, instance: ModelInstanceInfo, modified_body: str = None, modified_headers: str = None) -> Union[Response, StreamingResponse]:
    original_path = request.url.path
    path = original_path

    # Check if the path matches any in EMBEDDING_SUBPATHS and replace if necessary
    for subpath in EMBEDDING_SUBPATHS:
        if path.startswith(subpath["original"]):
            path = path.replace(subpath["original"], subpath["replacement"], 1)
            logger.info(f"Path replaced: {original_path} -> {path}")
            break

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
                raise HTTPException(status_code=500, detail="Internal server error")


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


def create_options_response(headers: dict) -> Response:
    options_headers = {
        "Access-Control-Allow-Origin": headers.get("Origin", "*"),
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        "Access-Control-Max-Age": "3600",
    }
    return Response(content="", status_code=204, headers=options_headers)


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


@manager_app.post("/add_model")
async def add_model(model_request: Union[ModelParser, List[ModelParser]]):
    logger.info(f"Received request to add model(s): {model_request}")

    if any(instance.status == "loading" for model in models.values() for instance in model.instances):
        logger.warning("Another model is currently loading. Request rejected.")
        raise HTTPException(status_code=409, detail="Another model is currently loading. Please try again later.")

    if isinstance(model_request, List):
        for model in model_request:
            await model_load_queue.put(model)
            logger.info(f"Model {model} instance queued for loading")
        return {"message": f"{len(model_request)} models queued for loading"}
    else:
        await model_load_queue.put(model_request)
        logger.info(f"Model {model_request} instance queued for loading")
        return {"message": f"Model {model_request} instance queued for loading"}


@manager_app.post("/remove_model")
async def remove_model(model_request: ModelRequest):
    if model_request.model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    for instance in models[model_request.model_id].instances:
        await stop_model_instance(model_request.model_id, instance.port)
    return {"message": f"All instances of model {model_request.model_id} removal initiated"}


@manager_app.get("/list_models")
async def list_models():
    return {name: {"instances": [{"port": inst.port, "status": inst.status} for inst in model.instances]}
            for name, model in models.items()}


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


async def modify_request(request: Request, changes: Dict[str, any]):
    body = await request.json()
    body_copy = deepcopy(body)

    # Apply the changes to the body
    for key, value in changes.items():
        if key in body_copy:
            body_copy[key] = value

    modified_body = json.dumps(body_copy).encode()

    # Create a copy of the headers and update the Content-Length
    modified_headers = dict(request.headers)
    modified_headers["content-length"] = str(len(modified_body))

    return modified_body, modified_headers


async def consolidate_responses(original_request: Request, responses: List[Response]):
    # This is a placeholder implementation. Replace with your actual consolidation logic.
    consolidation_prompt = "\n---\n" + textwrap.dedent("""
    Please synthesize the following reference answer into a single, high-quality response.
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
    """).strip() + "\n---\n"

    for index, response in enumerate(responses):
        response_body = json.loads(response.body)
        response_msg = response_body["choices"][0].get("message")

        # add the answer to the prompt
        consolidation_prompt += f"Reference answer {str(index+1)}\n"
        if response_msg and response_msg.get("content"):
            consolidation_prompt += response_msg["content"] + "\n"

        if response_msg and response_msg.get("tool_calls"):
            for tool_call in response_msg.get("tool_calls"):
                consolidation_prompt += str(tool_call["function"]) + "\n"

        consolidation_prompt += "---\n"

    consolidation_prompt += "Now provide the final synthesized response:\n"

    # Create a modified version of request with the answers from respective agents
    original_body = await original_request.json()

    modified_messages = original_body.get("messages", [])

    # append the final synthesis prompt to the message list
    modified_messages.append({
        "role": "user",
        "content": consolidation_prompt
    })

    modified_body, modified_headers = await modify_request(
        original_request,
        changes={"messages": modified_messages}
    )

    return modified_body, modified_headers


async def handle_mixture_of_agent_request(request: Request, body_json: dict):
    moa_config = MixtureOfAgents(**body_json.get("mixture_of_agents"))
    agent_list = moa_config.agent_list
    master_agent = body_json.get("model")

    if not agent_list or not master_agent:
        raise HTTPException(status_code=400, detail="Invalid request: missing agent_list or master_agent")

    # Validate all models upfront
    all_models = [agent.model if isinstance(agent, AgentWithThinking) else agent for agent in agent_list] + [master_agent]

    # Create a modified request where streaming is turned off
    changes = {"stream": False}
    modified_body, modified_headers = await modify_request(request, changes)

    # Forward the request to all models in the agent_list
    responses = await forward_to_multiple_agents(request, agent_list, modified_body, modified_headers)

    # Consolidate responses
    consol_modified_body, consol_modified_headers = await consolidate_responses(request, responses)

    # Forward consolidated response to master_agent
    instance = await get_instance_for_model(master_agent)
    final_response = await forward_request(request, instance, consol_modified_body, consol_modified_headers)
    return final_response


async def load_balanced_router(request: Request, path: str):
    if request.url.path in EXCLUDED_ENDPOINTS:
        return await request.app.router.get_route_handler()(request)

    if request.method == "OPTIONS":
        return create_options_response(dict(request.headers))

    body = await request.body()
    body_json = json.loads(body)

    if body_json.get("mixture_of_agents", False):
        return await handle_mixture_of_agent_request(request, body_json)

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
            available_instances = []

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

    # Ensure tasks are cancelled when the server stops
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

