from .model_manager import ModelManager
from .data_classes import VideoFrameCollection

model_manager = ModelManager()
video_frames = VideoFrameCollection()

def get_model_manager():
    if model_manager is None:
        raise RuntimeError("ModelManager not initialized")
    return model_manager

def get_video_collection():
    return video_frames