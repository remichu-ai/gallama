from typing import Optional
from .data_classes import ServerSetting
from .model_manager import ModelManager

model_manager: Optional[ModelManager] = None

def get_model_manager():
    if model_manager is None:
        raise RuntimeError("ModelManager not initialized")
    return model_manager

def initialize_model_manager(settings: ServerSetting):
    global model_manager
    model_manager = ModelManager()
    for model_spec in settings.model_specs:
        model_manager.load_model(model_spec)
    return model_manager