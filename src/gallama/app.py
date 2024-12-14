import torch.multiprocessing as mp
from fastapi import FastAPI, APIRouter
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from .config.config_manager import ConfigManager
from .logger.logger import get_logger
from typing import List, Dict
from .data_classes import ModelSpec, ServerSetting
from pydantic import BaseModel, Field
import asyncio
from .api_response import chat_response
from .dependencies import initialize_model_manager, get_model_manager
from .routes import chat
from contextlib import asynccontextmanager


logger = get_logger()

config_manager = ConfigManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model manager
    logger.info("Starting up server...")
    # model_manager = initialize_model_manager(settings)
    yield
    # Shutdown: cleanup
    logger.info("Shutting down server...")
    model_manager = get_model_manager()
    if model_manager:
        model_manager.cleanup()




# Set spawn method at the very beginning of the program, CUDA need spawn method
mp.set_start_method('spawn', force=True)


def make_server(settings: ServerSetting):
    app = FastAPI(lifespan=lifespan)

    # add middleware
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    initialize_model_manager(settings)

    # Include routers
    app.include_router(chat.router)
    # app.include_router(completion.router)
    # app.include_router(embeddings.router)

    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)

if __name__ == "__main__":
    mp.freeze_support()
    # uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)
