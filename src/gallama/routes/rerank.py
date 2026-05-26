from fastapi import APIRouter, HTTPException

from ..data_classes import RerankRequest, RerankResponse, RerankResult
from ..dependencies import get_model_manager
from ..logger import logger

router = APIRouter(prefix="/v1", tags=["rerank"])


@router.post("/rerank")
async def rerank(request: RerankRequest):
    model_manager = get_model_manager()
    reranker_model = model_manager.get_model(request.model, _type="reranker")
    if reranker_model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Reranker model not loaded: {request.model}",
        )

    results = await reranker_model.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
    )

    response_results = []
    for idx, score in results:
        result_kwargs = {"index": idx, "relevance_score": score}
        if request.return_documents and idx < len(request.documents):
            result_kwargs["document"] = {"text": request.documents[idx]}
        response_results.append(RerankResult(**result_kwargs))

    return RerankResponse(
        model=reranker_model.model_name,
        results=response_results,
    )
