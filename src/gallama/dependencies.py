from .model_manager import ModelManager
from .data_classes import VideoFrameCollection
from .data_classes.realtime_client_proto import SessionConfig

model_manager = ModelManager()
video_frames = VideoFrameCollection()


session_config = SessionConfig()

def get_model_manager():
    if model_manager is None:
        raise RuntimeError("ModelManager not initialized")
    return model_manager

def get_video_collection():
    return video_frames

async def get_session_config():
    return session_config

async def dependency_update_session_config(updated_session_config):
    global session_config
    session_config = updated_session_config