import hashlib
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _sanitize_mcp_name_part(value: str) -> str:
    """Normalize a string into a safe MCP name component.

    This removes leading/trailing whitespace, replaces non-alphanumeric
    characters with underscores, collapses repeated underscores, strips
    underscores from the ends, and returns "tool" if nothing remains.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "tool"


def make_mcp_synthetic_tool_name(server_name: str, tool_name: str) -> str:
    """
    standardize tool name from mcp to mcp__{server_part}__{tool_part}
    as mcp tool will appear as one of the tool as non mcp tool, this help differentiate
    in the event of colision with a tool name from non mcp tool, use the hash value
    """
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
    """Configuration for connecting to and filtering tools from a remote MCP server."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Logical name used to identify the remote MCP server in configuration and traces.")
    url: str = Field(description="Base URL used to connect to the remote MCP server.")
    authorization_token: Optional[str] = Field(
        default=None,
        description="Bearer token to send when the server requires authenticated requests.",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to include when connecting to the remote MCP server.",
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="Optional allowlist of remote tool names that may be exposed from this server.",
    )
    require_approval: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Optional approval policy controlling whether tool calls from this server need user confirmation.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary server-specific metadata preserved alongside the configuration.",
    )


class MCPResolvedTool(BaseModel):
    """Resolved metadata for a remote MCP tool after discovery from a configured server."""

    server_name: str = Field(description="Name of the remote MCP server that exposes this tool.")
    tool_name: str = Field(description="Original tool name returned by the remote MCP server.")
    synthetic_name: str = Field(description="Locally generated tool name used to uniquely reference the remote MCP tool.")
    description: Optional[str] = Field(default=None, description="Human-readable description of what the remote MCP tool does.")
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="JSON schema describing the arguments accepted by the remote MCP tool.",
    )


class MCPCallTraceError(BaseModel):
    """Structured error details captured for a failed remote MCP tool invocation."""

    message: str = Field(description="Human-readable error message returned or generated for a failed MCP tool call.")
    code: Optional[str] = Field(default=None, description="Optional machine-readable error code associated with the failure.")
    details: Optional[Any] = Field(default=None, description="Optional structured error payload with additional failure details.")


class MCPCallTrace(BaseModel):
    """Execution trace for a single remote MCP tool call, including inputs, output, and failure data."""

    call_id: str = Field(description="Unique identifier assigned to a single MCP tool invocation.")
    server_name: str = Field(description="Name of the remote MCP server that handled the tool invocation.")
    tool_name: str = Field(description="Original remote MCP tool name that was invoked.")
    synthetic_name: str = Field(description="Local synthetic tool name used when routing the invocation.")
    arguments_json: str = Field(description="Serialized JSON arguments that were sent to the remote MCP tool.")
    output_text: Optional[str] = Field(default=None, description="Normalized textual representation of the tool result, when available.")
    error: Optional[MCPCallTraceError] = Field(default=None, description="Error details captured when the MCP tool call fails or returns an error.")
