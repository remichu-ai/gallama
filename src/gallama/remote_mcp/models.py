import hashlib
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _sanitize_mcp_name_part(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "tool"


def make_mcp_synthetic_tool_name(server_name: str, tool_name: str) -> str:
    server_part = _sanitize_mcp_name_part(server_name)
    tool_part = _sanitize_mcp_name_part(tool_name)
    base_name = f"mcp__{server_part}__{tool_part}"

    # Keep names stable but bounded if sanitization collapses distinct names.
    raw_name = f"{server_name}\0{tool_name}"
    suffix = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    if len(base_name) <= 56:
        return base_name
    return f"{base_name[:56]}__{suffix}"


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    url: str
    authorization_token: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    allowed_tools: Optional[List[str]] = None
    require_approval: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPResolvedTool(BaseModel):
    server_name: str
    tool_name: str
    synthetic_name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})


class MCPCallTraceError(BaseModel):
    message: str
    code: Optional[str] = None
    details: Optional[Any] = None


class MCPCallTrace(BaseModel):
    call_id: str
    server_name: str
    tool_name: str
    synthetic_name: str
    arguments_json: str
    output_text: Optional[str] = None
    error: Optional[MCPCallTraceError] = None
