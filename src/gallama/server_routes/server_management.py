from fastapi import APIRouter
import httpx
import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from typing import Union
from gallama.data_classes.data_class import ModelSpec, ModelObjectResponse, ModelObject, ModelDownloadSpec
from gallama.data_classes import ModelRequest, StopModelByPort
from gallama.server_engine import download_model_from_hf
from typing import List, Dict
from gallama.config import ConfigManager
import psutil
import asyncio
from gallama.server_engine import log_model_status
from gallama.dependencies_server import get_server_manager, get_logger


logger = get_logger()

router = APIRouter(prefix="", tags=["server_management"])


# Add a health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy"}


@router.post("/download_model")
def download_model(model_spec: ModelDownloadSpec):
    download_model_from_hf(model_spec)


async def load_models(model_request, task_id):
    server_manager = get_server_manager()

    server_manager.task_status[task_id] = "loading"
    try:
        models_to_load = model_request if isinstance(model_request, List) else [model_request]
        for model in models_to_load:
            await server_manager.model_load_queue.put(model)
            logger.info(f"Model {model} instance queued for loading")

        # Wait for all models to be loaded
        await server_manager.model_load_queue.join()

        server_manager.task_status[task_id] = "completed"
    except Exception as e:
        server_manager.task_status[task_id] = f"failed: {str(e)}"

@router.post("/add_model")
@router.post("/v1/add_model")
async def add_model(model_request: Union[ModelSpec, List[ModelSpec]], background_tasks: BackgroundTasks, request: Request):
    server_manager = get_server_manager()

    task_id = str(uuid.uuid4())
    # Log the incoming request body
    logger.info(f"Received raw request: {await request.body()}")


    logger.info(f"Received request to add model(s): {model_request}, Task ID: {task_id}")

    if any(instance.status == "loading" for model in server_manager.models.values() for instance in model.instances):
        logger.warning("Another model is currently loading. Request rejected.")
        raise HTTPException(status_code=409, detail="Another model is currently loading. Please try again later.")

    server_manager.task_status[task_id] = "queued"
    background_tasks.add_task(load_models, model_request, task_id)

    return {"message": f"Model(s) queued for loading", "task_id": task_id}



@router.post("/v1/stop_model_by_port")
@router.post("/stop_model_by_port")
async def stop_model_by_port(request: StopModelByPort):
    server_manager = get_server_manager()

    model_to_remove = None
    instance_to_remove = None

    # Find the model and instance by port
    for model, model_data in server_manager.models.items():
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
        server_manager.models[model_to_remove].instances = [
            inst for inst in server_manager.models[model_to_remove].instances if inst.port != request.port
        ]

        # If no instances remain, remove the model entirely
        if not server_manager.models[model_to_remove].instances:
            del server_manager.models[model_to_remove]

        logger.info(f"Cleaned up instance on port {request.port}")
        log_model_status(server_manager.models, custom_logger=logger)  # Log status after removal


# Add a periodic health check function
async def periodic_health_check():
    while True:
        await asyncio.sleep(60)  # Check every minute
        server_manager = get_server_manager()
        for model_name, model_info in server_manager.models.items():
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

async def stop_model_instance(model: str, port: int):
    server_manager = get_server_manager()

    if model not in server_manager.models:
        logger.warning(f"Attempted to stop instance of non-existent model: {model}")
        return

    instance = next((inst for inst in server_manager.models[model].instances if inst.port == port), None)
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
        server_manager.models[model].instances = [inst for inst in server_manager.models[model].instances if inst.port != port]
        if not server_manager.models[model].instances:
            del server_manager.models[model]
        logger.info(f"Cleaned up model {model} instance on port {port}")
        log_model_status(server_manager.models, custom_logger=logger)   # Log status after removing a model instance


@router.get("/task_status/{task_id}")
@router.get("/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    server_manager = get_server_manager()
    status = server_manager.task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": status}


@router.post("/remove_model")
@router.post("/v1/remove_model")
async def remove_model(model_request: ModelRequest):
    server_manager = get_server_manager()

    if model_request.model_id not in server_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    for instance in server_manager.models[model_request.model_id].instances:
        await stop_model_instance(model_request.model_id, instance.port)
    return {"message": f"All instances of model {model_request.model_id} removal initiated"}



@router.get("/v1/list_loaded_models")
@router.get("/list_loaded_models")
async def list_models():
    server_manager = get_server_manager()

    return {name: {"instances": [{"port": inst.port, "status": inst.status} for inst in model.instances]}
            for name, model in server_manager.models.items()}


@router.get("/list_available_models")
@router.get("/v1/list_available_models")
async def list_available_models():
    config_manager = ConfigManager()
    downloaded_models = config_manager.list_downloaded_models_dict
    # logger.info(downloaded_models)
    return downloaded_models


@router.get("/v1/models")
async def get_models(request: Request):
    server_manager = get_server_manager()

    data = []
    for name, model in server_manager.models.items():
        data.append(ModelObject(id=name))

    return ModelObjectResponse(data=data)


@router.get("/loading_status")
async def loading_status():
    server_manager = get_server_manager()

    return {
        "queue_size": server_manager.model_load_queue.qsize(),
        "currently_loading": server_manager.loading_lock.locked(),
        "models": {name: {"instances": [{"port": inst.port, "status": inst.status} for inst in model.instances]}
                   for name, model in server_manager.models.items()}
    }