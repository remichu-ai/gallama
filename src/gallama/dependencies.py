from .model_manager import ModelManager
from .data_classes import VideoFrameCollection
from .data_classes.realtime_client_proto import SessionConfig

model_manager = ModelManager()
video_frames = VideoFrameCollection()
record_start_time: float = 0    # track the time when user click start recoridng

clear_history_flag: bool = False  # clear LLM past history

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

def get_record_start_time():
    global record_start_time
    return record_start_time

def set_record_start_time(new_timing: float):
    global record_start_time
    record_start_time = new_timing

def get_clear_history_flag():
    global clear_history_flag
    return clear_history_flag

def set_clear_history_flag(new_flag: float):
    global clear_history_flag
    clear_history_flag = new_flag