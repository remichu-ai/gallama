from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator
from typing import Optional, Literal, List, Dict, Union, Any, Type


class ModelRequest(BaseModel):
    model_id: str

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInstanceInfo(BaseModel):
    model_name: str
    port: int
    pid: int  # Changed from process to pid
    status: str
    model_type: Literal["stt", "llm", "tts", "embedding"]
    strict: bool = Field(description="whether require api call to match model name or not", default=False)
    max_concurrent_requests: int = Field(
        description="number of concurrent requests the instance is expected to handle",
        default=1,
    )

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInfo(BaseModel):
    instances: List[ModelInstanceInfo]

class StopModelByPort(BaseModel):
    port: int
