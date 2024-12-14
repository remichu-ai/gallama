from .model_manager import ModelManager

model_manager = ModelManager()

def get_model_manager():
    if model_manager is None:
        raise RuntimeError("ModelManager not initialized")
    return model_manager