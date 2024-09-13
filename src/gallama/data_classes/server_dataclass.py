from pydantic import BaseModel, Field, validator, ConfigDict, RootModel, field_validator, constr, model_validator
from typing import Optional, Literal, List, Dict, Union, Any, Type


class ModelRequest(BaseModel):
    model_id: str

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInstanceInfo(BaseModel):
    model_id: str
    port: int
    pid: int  # Changed from process to pid
    status: str
    embedding: bool = False

    class Config:
        # Override protected namespaces to suppress warning
        protected_namespaces = ()


class ModelInfo(BaseModel):
    instances: List[ModelInstanceInfo]


class AgentWithThinking(BaseModel):
    model: str = Field(description="The name of the model")
    thinking_template: Optional[str] = Field(default=None, description="The XML thinking for the agent")


class MixtureOfAgents(BaseModel):
    agent_list: List[Union[str, AgentWithThinking]]

class StopModelByPort(BaseModel):
    port: int