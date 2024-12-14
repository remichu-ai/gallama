from fastapi import Request, APIRouter
from gallama.data_classes import (
    ModelObjectResponse,
    ModelObject,
    ModelSpec
)
from ..logger import logger
from ..dependencies import get_model_manager

router = APIRouter(prefix="", tags=["model_management"])


@router.post("/load_model")
def load_model(model_spec: ModelSpec):
    """
    model_spec is model specification coming from cli
    it might not have all the properties required for the model to be loaded
    the config_manager below contain all the models properties
    """
    model_manager = get_model_manager()
    model_manager.load_model(model_spec)


@router.get("/v1/models")
async def get_models(request: Request):
    model_manager = get_model_manager()

    data = []
    for model_name in model_manager.llm_dict.keys():
        data.append(ModelObject(id=model_name))
    for model_name in model_manager.stt_dict.keys():
        data.append(ModelObject(id=model_name))
    for model_name in model_manager.tts_dict.keys():
        data.append(ModelObject(id=model_name))
    for model_name in model_manager.embedding_dict.keys():
        data.append(ModelObject(id=model_name))
    return ModelObjectResponse(data=data)

# @router.post("/delete_model")
# def delete_model(model_name: str):
#
#     if model_name in llm_dict.keys():
#         del llm_dict[model_name]
#
#         # Clear CUDA cache if using GPU
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         # Run garbage collection
#         gc.collect()
#
#         logger.info("Deleted model: " + model_name)


@router.get("/status")
async def get_status():
    model_manager = get_model_manager()
    logger.info("Status endpoint called")
    return {"status": "ready" if model_manager.model_ready else "loading"}

# Add a health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy"}