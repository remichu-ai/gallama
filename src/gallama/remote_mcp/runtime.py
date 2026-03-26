import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from .models import (
    MCPCallTrace,
    MCPCallTraceError,
    MCPResolvedTool,
    MCPServerConfig,
    make_mcp_synthetic_tool_name,
)


class MCPRuntimeError(RuntimeError):
    pass


class MCPRuntime:
    def _build_headers(self, server: MCPServerConfig) -> Dict[str, str]:
        headers = dict(server.headers or {})
        normalized_keys = {key.lower() for key in headers}
        if server.authorization_token and "authorization" not in normalized_keys:
            headers["Authorization"] = f"Bearer {server.authorization_token}"
        return headers

    @asynccontextmanager
    async def _session(self, server: MCPServerConfig) -> AsyncIterator[ClientSession]:
        headers = self._build_headers(server)
        errors: List[Exception] = []

        transports = (
            ("sse", sse_client(server.url, headers=headers)),
        )

        try:
            async with httpx.AsyncClient(headers=headers) as client:
                async with streamable_http_client(server.url, http_client=client) as streams:
                    read_stream, write_stream = streams[:2]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
                        return
        except Exception as exc:  # pragma: no cover - exercised through runtime fallback
            errors.append(exc)

        for _transport_name, client_cm in transports:
            try:
                async with client_cm as streams:
                    read_stream, write_stream = streams[:2]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
                        return
            except Exception as exc:  # pragma: no cover - exercised through runtime fallback
                errors.append(exc)

        error_messages = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors) or "unknown transport error"
        raise MCPRuntimeError(
            f"Unable to connect to MCP server '{server.name}' at {server.url}: {error_messages}"
        )

    async def list_tools(self, server: MCPServerConfig) -> List[MCPResolvedTool]:
        async with self._session(server) as session:
            result = await session.list_tools()

        allowed = set(server.allowed_tools or [])
        resolved_tools: List[MCPResolvedTool] = []
        for tool in result.tools:
            if allowed and tool.name not in allowed:
                continue

            input_schema = tool.inputSchema or {"type": "object", "properties": {}, "required": []}
            resolved_tools.append(
                MCPResolvedTool(
                    server_name=server.name,
                    tool_name=tool.name,
                    synthetic_name=make_mcp_synthetic_tool_name(server.name, tool.name),
                    description=tool.description,
                    input_schema=input_schema,
                )
            )

        return resolved_tools

    async def call_tool(self, server: MCPServerConfig, tool: MCPResolvedTool, arguments: Dict[str, Any], call_id: str) -> MCPCallTrace:
        try:
            async with self._session(server) as session:
                result = await session.call_tool(tool.tool_name, arguments=arguments)
        except Exception as exc:
            return MCPCallTrace(
                call_id=call_id,
                server_name=server.name,
                tool_name=tool.tool_name,
                synthetic_name=tool.synthetic_name,
                arguments_json=json.dumps(arguments, ensure_ascii=False),
                error=MCPCallTraceError(message=str(exc)),
            )

        return MCPCallTrace(
            call_id=call_id,
            server_name=server.name,
            tool_name=tool.tool_name,
            synthetic_name=tool.synthetic_name,
            arguments_json=json.dumps(arguments, ensure_ascii=False),
            output_text=self.result_to_tool_message(result),
            error=None if not result.isError else MCPCallTraceError(message=self.result_to_tool_message(result)),
        )

    @staticmethod
    def result_to_tool_message(result: CallToolResult) -> str:
        text_parts: List[str] = []
        rich_parts: List[Any] = []

        for item in result.content or []:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
            elif isinstance(item, ImageContent):
                rich_parts.append({"type": "image", "mime_type": item.mimeType, "data": item.data})
            elif isinstance(item, AudioContent):
                rich_parts.append({"type": "audio", "mime_type": item.mimeType, "data": item.data})
            elif isinstance(item, ResourceLink):
                rich_parts.append(item.model_dump(mode="json", by_alias=True))
            elif isinstance(item, EmbeddedResource):
                resource = item.resource
                if isinstance(resource, TextResourceContents):
                    text_parts.append(resource.text)
                elif isinstance(resource, BlobResourceContents):
                    rich_parts.append(resource.model_dump(mode="json", by_alias=True))
                else:
                    rich_parts.append(item.model_dump(mode="json", by_alias=True))
            else:
                rich_parts.append(item.model_dump(mode="json", by_alias=True))

        if result.structuredContent is not None:
            rich_parts.append({"structured_content": result.structuredContent})

        if rich_parts:
            serialized = json.dumps(rich_parts, ensure_ascii=False)
            if text_parts:
                return "\n\n".join(["\n".join(text_parts), serialized])
            return serialized

        if text_parts:
            return "\n".join(text_parts)

        return json.dumps(result.model_dump(mode="json", by_alias=True), ensure_ascii=False)
