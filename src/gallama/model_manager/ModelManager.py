import torch.multiprocessing as mp
from fastapi import FastAPI, APIRouter
from ..config.config_manager import ConfigManager
from ..logger.logger import get_logger
from typing import List, Dict
from ..data_classes import ModelSpec, ServerSetting, GenQueue
from pydantic import BaseModel, Field
from dataclasses import dataclass
import asyncio
logger = get_logger()
router = APIRouter()
config_manager = ConfigManager()

@dataclass
class ModelRequest:
    """ this class should encompass all the info needed for the model to process the request"""
    type: str  # 'chat', 'completion', 'embedding', etc.
    data: dict
    response_queue: mp.Queue     # queue to return the generated data from model
    stream: bool = False

class ModelProcess:
    def __init__(self, model_spec: ModelSpec):
        self.model_spec = model_spec
        self.request_queue = mp.Queue()
        self.process = None

    def start(self):
        self.process = mp.Process(
            target=self._run_model,
            args=(self.model_spec, self.request_queue)
        )
        self.process.start()

    @staticmethod
    def _run_model(model_spec: ModelSpec, request_queue: mp.Queue):

        # Load model configuration
        model_name = model_spec.model_name
        model_config = config_manager.get_model_config(model_name)

        # Merge configurations using the new method
        model_spec = ModelSpec.from_merged_config(model_spec, model_config)

        logger.info(f"Started process for model: {model_spec.model_name}")


        if not model_config:
            raise Exception(f"Model config for '{model_name}' not exist in ~/gallama/model_config.yaml")

        # Import model backends only within the process
        logger.info(f"model_spec: {model_spec}")
        logger.info(f"model_config: {model_config}")
        if model_spec.backend == 'exllama':
            from ..backend.llm import ModelExllama as ModelClass
        elif model_spec.backend == 'llama_cpp':
            from ..backend.llm import ModelTransformers as ModelClass
        elif model_spec.backend == "transformers":
            from ..backend.llm import ModelLlamaCpp as ModelClass
        elif model_spec.backend == "embedding":
            from ..backend.embedding import EmbeddingModel as ModelClass

        try:


            # model_class = model_class_dict.get(model_config["backend"])
            if not ModelClass:
                raise ValueError(f"Unsupported backend: {model_config['backend']}")

            # Initialize model
            model = ModelClass(model_spec=model_spec)

            logger.info(f"Model {model_name} loaded successfully")

            # Process generation requests
            while True:
                request: ModelRequest = request_queue.get()  # Blocks until request arrives

                if request is None:  # Shutdown signal
                    break

                request.data["gen_queue"] = request_queue

                try:
                    if request.type == "chat":
                        response = model.chat(**request.data)
                    elif request.type == "completion":
                        response = model.chat_raw(**request.data)
                    elif request.type == "embedding":
                        response = model.text_embeddings(**request.data)
                    else:
                        raise ValueError(f"Unknown request type: {request.type}")

                    request.gen_queue.put(("success", response))
                except Exception as e:
                    logger.error(f"Error processing request: {str(e)}")
                    request.gen_queue.put(("error", str(e)))

        except Exception as e:
            logger.error(f"Error in model process: {str(e)}")
        finally:
            logger.info(f"Shutting down model process for {model_spec.model_id}")


class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelProcess] = {}               # dict to all model process object
        self.model_semaphores: Dict[str, mp.Semaphore] = {}     # how many concurrent request a particular model can handle

    def load_model(self, model_spec: ModelSpec):
        model_process = ModelProcess(model_spec)
        model_process.start()

        # register to the class dict
        self.models[model_spec.model_name] = model_process
        self.model_semaphores[model_spec.model_name] = mp.Semaphore(model_spec.max_concurrent_requests)


    async def process_request(self, model_id: str, request: ModelRequest) -> GenQueue:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        gen_queue = GenQueue()

        # Create a background task to handle the request
        asyncio.create_task(
            self._handle_request(model_id, request, gen_queue)
        )

        return gen_queue

    async def _handle_request(self, model_id: str, request: ModelRequest, gen_queue: GenQueue):
        semaphore = self.model_semaphores[model_id]

        try:
            # Try to acquire semaphore with timeout
            acquired = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: semaphore.acquire(timeout=60)  # 60 second timeout
            )

            if not acquired:
                await gen_queue.put({"error": "Request timeout - model is busy"})
                return

            try:
                # Just send the request to model process and return the queue
                self.models[model_id].request_queue.put(request)

                if request.stream:
                    while True:
                        # Get generation events directly from response_queue
                        result = request.response_queue.get()
                        if result:
                            # Forward the generation event to the client
                            await gen_queue.put(result)

                else:
                    result = request.response_queue.get()
                    await gen_queue.put(result)

                return request.response_queue
            finally:
                semaphore.release()

        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            await gen_queue.put({"error": str(e)})

    def cleanup(self):
        """Cleanup all model processes"""
        for model_id, model_process in self.models.items():
            logger.info(f"Shutting down model: {model_id}")
            model_process.request_queue.put(None)  # Send shutdown signal
            model_process.process.join(timeout=5)  # Wait up to 5 seconds
            if model_process.process.is_alive():
                model_process.process.terminate()
