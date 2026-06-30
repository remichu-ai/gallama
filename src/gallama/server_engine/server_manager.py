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
        self.log_file: str | None = None

    def resolve_model_name(self, model_name: Optional[str]) -> Optional[str]:
        if model_name is None:
            return None
        if model_name in self.models:
            return model_name

        for canonical_name, model_info in self.models.items():
            for instance in model_info.instances:
                if model_name in instance.aliases:
                    return canonical_name

        return None

    def list_model_ids(self) -> List[str]:
        model_ids = []
        seen = set()

        for model_name, model_info in self.models.items():
            if model_name not in seen:
                model_ids.append(model_name)
                seen.add(model_name)
            for instance in model_info.instances:
                for alias in instance.aliases:
                    if alias not in seen:
                        model_ids.append(alias)
                        seen.add(alias)

        return model_ids

    def get_instance(
        self,
        model_type: Literal["stt", "llm", "tts", "embedding", "reranker"],
        model_name: Optional[str] = None
    ) -> Optional[ModelInstanceInfo]:
        resolved_model_name = self.resolve_model_name(model_name)

        # Iterate through all models to find a matching instance
        for model_info in self.models.values():
            for instance in model_info.instances:
                if instance.model_type == model_type:
                    if model_name is None or instance.model_name == resolved_model_name:
                        return instance
                    elif not instance.strict:
                        return instance

        # If no matching instance is found, return None
        return None
