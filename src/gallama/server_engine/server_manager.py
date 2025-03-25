from gallama.data_classes import ModelInfo, ModelInstanceInfo
from typing import List, Dict, Literal, Optional
import asyncio


class ServerManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}               # dict to all llm models process object
        self.model_load_queue = asyncio.Queue()
        self.loading_lock = asyncio.Lock()
        self.active_requests_lock = asyncio.Lock()
        self.task_status = {}

    def get_instance(
        self,
        model_type: Literal["stt", "llm", "tts", "embedding"],
        model_name: Optional[str] = None
    ) -> Optional[ModelInstanceInfo]:
        # Iterate through all models to find a matching instance
        for model_info in self.models.values():
            for instance in model_info.instances:
                if instance.model_type == model_type:
                    if model_name is None or instance.model_name == model_name:
                        return instance
                    elif not instance.strict:
                        return instance

        # If no matching instance is found, return None
        return None