from fastapi import APIRouter
from .chat import router as chat_router
from .embedding import router as embedding_router
from .model_management import router as model_management_router
from .audio import router as audio_router
from .ws_stt import router as ws_stt_router
from .ws_llm import router as ws_llm_router
from .ws_tts import router as ws_tts_router