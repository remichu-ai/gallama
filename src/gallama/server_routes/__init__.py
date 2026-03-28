from fastapi import APIRouter
from .server_management import router as server_management_router
from .server_management import periodic_health_check, stop_model_instance, download_model

responses_ws_router = APIRouter()
try:
    from .responses_ws import router as responses_ws_router
except ImportError:
    pass

realtime_router = APIRouter()
try:
    from .realtime import router as realtime_router
except ImportError:
    pass
