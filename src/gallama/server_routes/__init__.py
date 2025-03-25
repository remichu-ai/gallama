from .server_management import router as server_management_router
from .server_management import periodic_health_check, stop_model_instance, download_model
from .realtime import router as realtime_router