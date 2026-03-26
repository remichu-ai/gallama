import asyncio
import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import HTTPException, Request

from ..api_response.chat_response import chat_completion_response
from ..data_classes import (
    AnthropicMessagesResponse,
    BaseMessage,
    ChatCompletionResponse,
    ChatMLQuery,
    FunctionCall,
    GenQueueDynamic,
    ParameterSpec,
    ResponseMCPCallItem,
    ResponseMCPListToolsItem,
    ResponsesCreateResponse,
    ToolCall,
    ToolSpec,
    response_output_to_assistant_messages,
)
from ..data_classes.data_class import (
    AnthropicMCPToolResultBlock,
    AnthropicMCPToolUseBlock,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    FunctionSpec,
)
from ..remote_mcp.models import MCPCallTrace, MCPResolvedTool, MCPServerConfig
from ..request_validation import validate_api_request
from .runtime import MCPRuntime


def _clone_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    return [message.model_copy(deep=True) for message in messages]


def _mcp_tool_spec(tool: MCPResolvedTool) -> ToolSpec:
    schema = tool.input_schema or {"type": "object", "properties": {}, "required": []}
    description = tool.description or ""
    prefix = f"[Remote MCP server: {tool.server_name}; tool: {tool.tool_name}]"
    if description:
        description = f"{prefix} {description}"
    else:
        description = prefix

    return ToolSpec(
        type="function",
        function=FunctionSpec(
            name=tool.synthetic_name,
            description=description,
            parameters=ParameterSpec(
                type=schema.get("type", "object"),
                properties=schema.get("properties"),
                required=schema.get("required", []),
            ),
        ),
    )


def _openai_response_to_assistant_messages(response: ChatCompletionResponse) -> List[BaseMessage]:
    if not response.choices:
        return []

    message = response.choices[0].message
    tool_calls = [
        ToolCall(
            id=tool_call.id or "",
            function=FunctionCall(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            ),
            type=tool_call.type or "function",
            index=tool_call.index,
        )
        for tool_call in (message.tool_calls or [])
    ] or None

    reasoning = getattr(message, "reasoning", None) or getattr(message, "thinking", None)
    return [
        BaseMessage(
            role="assistant",
            content=message.content or "",
            tool_calls=tool_calls,
            reasoning=reasoning,
        )
    ]


def _anthropic_response_to_assistant_messages(response: AnthropicMessagesResponse) -> List[BaseMessage]:
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for block in response.content:
        if isinstance(block, AnthropicTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, AnthropicThinkingBlock):
            reasoning_parts.append(block.thinking)
        elif isinstance(block, AnthropicToolUseBlock):
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=FunctionCall(
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    ),
                    type="function",
                )
            )

    return [
        BaseMessage(
            role="assistant",
            content="".join(text_parts),
            tool_calls=tool_calls or None,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
        )
    ]


def provider_response_to_assistant_messages(
    provider: Literal["openai", "responses", "anthropic"],
    response_obj: Any,
) -> List[BaseMessage]:
    if provider == "openai":
        return _openai_response_to_assistant_messages(response_obj)
    if provider == "responses":
        return response_output_to_assistant_messages(response_obj.output)
    return _anthropic_response_to_assistant_messages(response_obj)


def _prepend_mcp_traces_to_responses(
    response_obj: ResponsesCreateResponse,
    discovered_tools: Dict[str, List[MCPResolvedTool]],
    call_traces: List[MCPCallTrace],
) -> ResponsesCreateResponse:
    extra_output: List[Any] = []
    for server_name, tools in discovered_tools.items():
        extra_output.append(
            ResponseMCPListToolsItem(
                server_label=server_name,
                tools=[
                    {
                        "name": tool.tool_name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools
                ],
            )
        )

    for trace in call_traces:
        extra_output.append(
            ResponseMCPCallItem(
                id=trace.call_id,
                arguments=trace.arguments_json,
                error=trace.error.model_dump(mode="json") if trace.error else None,
                name=trace.tool_name,
                output=trace.output_text,
                server_label=trace.server_name,
            )
        )

    if extra_output:
        response_obj.output = extra_output + list(response_obj.output)
    return response_obj


def _prepend_mcp_traces_to_anthropic(
    response_obj: AnthropicMessagesResponse,
    call_traces: List[MCPCallTrace],
) -> AnthropicMessagesResponse:
    extra_blocks: List[Any] = []
    for trace in call_traces:
        extra_blocks.append(
            AnthropicMCPToolUseBlock(
                id=trace.call_id,
                name=trace.tool_name,
                server_name=trace.server_name,
                input=json.loads(trace.arguments_json),
            )
        )
        result_text = trace.output_text or (trace.error.message if trace.error else "")
        extra_blocks.append(
            AnthropicMCPToolResultBlock(
                tool_use_id=trace.call_id,
                is_error=trace.error is not None,
                content=[AnthropicTextBlock(text=result_text)],
            )
        )

    if extra_blocks:
        response_obj.content = extra_blocks + list(response_obj.content)
    return response_obj


def prepend_mcp_traces_to_response(
    provider: Literal["openai", "responses", "anthropic"],
    response_obj: Any,
    discovered_tools: Dict[str, List[MCPResolvedTool]],
    call_traces: List[MCPCallTrace],
) -> Any:
    if provider == "responses":
        return _prepend_mcp_traces_to_responses(response_obj, discovered_tools, call_traces)
    if provider == "anthropic":
        return _prepend_mcp_traces_to_anthropic(response_obj, call_traces)
    return response_obj


class MCPLoopResult:
    def __init__(self, response_obj: Any, conversation_messages: List[BaseMessage]):
        self.response_obj = response_obj
        self.conversation_messages = conversation_messages


async def run_mcp_completion_loop(
    *,
    provider: Literal["openai", "responses", "anthropic"],
    base_query: ChatMLQuery,
    llm: Any,
    request: Request,
    conversation_messages: List[BaseMessage],
    mcp_servers: List[MCPServerConfig],
    formatter_kwargs: Optional[Dict[str, Any]] = None,
    max_iterations: int = 8,
) -> MCPLoopResult:
    if base_query.stream:
        raise HTTPException(status_code=400, detail="Streaming with MCP tools is not supported yet")

    runtime = MCPRuntime()
    function_tools, _ = base_query.split_tools()
    discovered_tools: Dict[str, List[MCPResolvedTool]] = {}
    resolved_by_synthetic_name: Dict[str, tuple[MCPServerConfig, MCPResolvedTool]] = {}
    merged_tools: List[ToolSpec] = list(function_tools)

    for server in mcp_servers:
        require_approval = server.require_approval
        if require_approval not in (None, "never"):
            raise HTTPException(
                status_code=400,
                detail=f"MCP require_approval for server '{server.name}' is not supported yet",
            )

        server_tools = await runtime.list_tools(server)
        discovered_tools[server.name] = server_tools
        for tool in server_tools:
            resolved_by_synthetic_name[tool.synthetic_name] = (server, tool)
            merged_tools.append(_mcp_tool_spec(tool))

    if not merged_tools:
        raise HTTPException(status_code=400, detail="No tools available after MCP discovery")

    working_messages = _clone_messages(conversation_messages)
    call_traces: List[MCPCallTrace] = []

    for _ in range(max_iterations):
        query_for_turn = validate_api_request(
            base_query.model_copy(
                deep=True,
                update={
                    "messages": _clone_messages(working_messages),
                    "tools": merged_tools,
                },
            )
        )

        gen_queue_dynamic = GenQueueDynamic()
        asyncio.create_task(
            llm.chat(
                query=query_for_turn,
                prompt_eng=llm.prompt_eng,
                gen_queue=gen_queue_dynamic,
                request=request,
            )
        )

        response_obj = await chat_completion_response(
            query=query_for_turn,
            gen_queue=gen_queue_dynamic,
            model_name=llm.model_name,
            request=request,
            tag_definitions=llm.prompt_eng.tag_definitions,
            provider=provider,
            formatter_kwargs=formatter_kwargs,
        )

        assistant_messages = provider_response_to_assistant_messages(provider, response_obj)
        if not assistant_messages:
            return MCPLoopResult(
                prepend_mcp_traces_to_response(provider, response_obj, discovered_tools, call_traces),
                working_messages,
            )

        working_messages.extend(_clone_messages(assistant_messages))
        latest_assistant = assistant_messages[-1]
        tool_calls = latest_assistant.tool_calls or []

        if not tool_calls:
            return MCPLoopResult(
                prepend_mcp_traces_to_response(provider, response_obj, discovered_tools, call_traces),
                working_messages,
            )

        mcp_tool_calls = [tool_call for tool_call in tool_calls if tool_call.function.name in resolved_by_synthetic_name]
        if len(mcp_tool_calls) != len(tool_calls):
            if mcp_tool_calls:
                raise HTTPException(
                    status_code=400,
                    detail="Mixed MCP and non-MCP tool calls in the same turn are not supported yet",
                )
            return MCPLoopResult(
                prepend_mcp_traces_to_response(provider, response_obj, discovered_tools, call_traces),
                working_messages,
            )

        if query_for_turn.parallel_tool_calls and len(mcp_tool_calls) > 1:
            executed_traces = await asyncio.gather(
                *[
                    runtime.call_tool(
                        server=resolved_by_synthetic_name[tool_call.function.name][0],
                        tool=resolved_by_synthetic_name[tool_call.function.name][1],
                        arguments=json.loads(tool_call.function.arguments or "{}"),
                        call_id=tool_call.id,
                    )
                    for tool_call in mcp_tool_calls
                ]
            )
        else:
            executed_traces = []
            for tool_call in mcp_tool_calls:
                server, resolved_tool = resolved_by_synthetic_name[tool_call.function.name]
                executed_traces.append(
                    await runtime.call_tool(
                        server=server,
                        tool=resolved_tool,
                        arguments=json.loads(tool_call.function.arguments or "{}"),
                        call_id=tool_call.id,
                    )
                )

        for trace in executed_traces:
            call_traces.append(trace)
            tool_content = trace.output_text
            if trace.error and not tool_content:
                tool_content = json.dumps(trace.error.model_dump(mode="json"), ensure_ascii=False)

            working_messages.append(
                BaseMessage(
                    role="tool",
                    tool_call_id=trace.call_id,
                    content=tool_content or "",
                )
            )

    raise HTTPException(status_code=400, detail="MCP tool loop exceeded maximum iterations")
