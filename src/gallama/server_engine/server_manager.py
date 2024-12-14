from gallama.data_classes import ModelInfo
from typing import List, Dict
import asyncio


class ServerManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}               # dict to all llm models process object
        self.model_load_queue = asyncio.Queue()
        self.loading_lock = asyncio.Lock()
        self.active_requests_lock = asyncio.Lock()
        self.task_status = {}
