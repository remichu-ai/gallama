from fastapi import APIRouter, Request, HTTPException
from ..dependencies import get_model_manager
from ..logger import logger
from gallama.data_classes import (
    EmbeddingRequest,
)

# https://platform.openai.com/docs/api-reference/embeddings/create

router = APIRouter(prefix="/v1", tags=["embedding"])

@router.post("/embeddings")
async def embeddings(request: Request, query: EmbeddingRequest):
    model_manager = get_model_manager()
    embedding_model = model_manager.get_model(query.model, _type="embedding")
    # embedding_model = model_manager.embedding_dict[query.model]

    # for embedding, hard enforcement of matching model name
    if query.model != embedding_model.model_name:
        logger.warning(f"Embedding model {query.model} is not found. Use current loaded model: {embedding_model.model_name}")
        # raise HTTPException(status_code=400,
        #                     detail=f"Embedding model {query.model} is not found. Current loaded model is {embedding_model.model_name}")

    return await embedding_model.text_embeddings(query=query)