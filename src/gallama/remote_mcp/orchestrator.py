import asyncio
from collections import deque
import json
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from fastapi import HTTPException, Request

from ..api_response.api_formatter import AnthropicFormatter, ResponsesFormatter
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
from ..data_classes.generation_data_class import GenQueue
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
from ..logger import logger
from .runtime import MCPRuntime, MCPRuntimeError


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


def _build_responses_mcp_list_tools_items(
    discovered_tools: Dict[str, List[MCPResolvedTool]],
) -> List[ResponseMCPListToolsItem]:
    items: List[ResponseMCPListToolsItem] = []
    for server_name, tools in discovered_tools.items():
        items.append(
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
    return items


def _build_responses_mcp_call_items(call_traces: List[MCPCallTrace]) -> List[ResponseMCPCallItem]:
    return [
        ResponseMCPCallItem(
            id=trace.call_id,
            arguments=trace.arguments_json,
            error=trace.error.model_dump(mode="json") if trace.error else None,
            name=trace.tool_name,
            output=trace.output_text,
            server_label=trace.server_name,
            status="failed" if trace.error else "completed",
        )
        for trace in call_traces
    ]


def _prepend_mcp_traces_to_responses(
    response_obj: ResponsesCreateResponse,
    discovered_tools: Dict[str, List[MCPResolvedTool]],
    call_traces: List[MCPCallTrace],
) -> ResponsesCreateResponse:
    extra_output: List[Any] = [
        *_build_responses_mcp_list_tools_items(discovered_tools),
        *_build_responses_mcp_call_items(call_traces),
    ]

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


async def _prepare_mcp_tools(
    *,
    base_query: ChatMLQuery,
    mcp_servers: List[MCPServerConfig],
) -> tuple[MCPRuntime, Dict[str, List[MCPResolvedTool]], Dict[str, tuple[MCPServerConfig, MCPResolvedTool]], List[ToolSpec]]:
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

        try:
            server_tools = await runtime.list_tools(server)
        except MCPRuntimeError as exc:
            logger.warning(
                "MCP server unavailable during tool discovery; skipping server '%s' at %s: %s",
                server.name,
                server.url,
                exc,
            )
            continue

        discovered_tools[server.name] = server_tools
        for tool in server_tools:
            resolved_by_synthetic_name[tool.synthetic_name] = (server, tool)
            merged_tools.append(_mcp_tool_spec(tool))

    if mcp_servers and not discovered_tools:
        logger.warning("All configured MCP servers were unavailable; proceeding without MCP tools")

    return runtime, discovered_tools, resolved_by_synthetic_name, merged_tools


def _build_query_for_turn(
    *,
    base_query: ChatMLQuery,
    working_messages: List[BaseMessage],
    merged_tools: List[ToolSpec],
) -> ChatMLQuery:
    return validate_api_request(
        base_query.model_copy(
            deep=True,
            update={
                "messages": _clone_messages(working_messages),
                "tools": merged_tools,
            },
        )
    )


def _partition_mcp_tool_calls(
    tool_calls: List[ToolCall],
    resolved_by_synthetic_name: Dict[str, tuple[MCPServerConfig, MCPResolvedTool]],
) -> List[ToolCall]:
    mcp_tool_calls = [tool_call for tool_call in tool_calls if tool_call.function.name in resolved_by_synthetic_name]

    if len(mcp_tool_calls) != len(tool_calls):
        if mcp_tool_calls:
            raise HTTPException(
                status_code=400,
                detail="Mixed MCP and non-MCP tool calls in the same turn are not supported yet",
            )
        return []

    return mcp_tool_calls


async def _execute_mcp_tool_calls(
    *,
    runtime: MCPRuntime,
    query_for_turn: ChatMLQuery,
    resolved_by_synthetic_name: Dict[str, tuple[MCPServerConfig, MCPResolvedTool]],
    tool_calls: List[ToolCall],
) -> List[MCPCallTrace]:
    if query_for_turn.parallel_tool_calls and len(tool_calls) > 1:
        return await asyncio.gather(
            *[
                runtime.call_tool(
                    server=resolved_by_synthetic_name[tool_call.function.name][0],
                    tool=resolved_by_synthetic_name[tool_call.function.name][1],
                    arguments=json.loads(tool_call.function.arguments or "{}"),
                    call_id=tool_call.id,
                )
                for tool_call in tool_calls
            ]
        )

    executed_traces = []
    for tool_call in tool_calls:
        server, resolved_tool = resolved_by_synthetic_name[tool_call.function.name]
        executed_traces.append(
            await runtime.call_tool(
                server=server,
                tool=resolved_tool,
                arguments=json.loads(tool_call.function.arguments or "{}"),
                call_id=tool_call.id,
            )
        )
    return executed_traces


class MCPLoopResult:
    def __init__(self, response_obj: Any, conversation_messages: List[BaseMessage]):
        self.response_obj = response_obj
        self.conversation_messages = conversation_messages


class MCPStreamController:
    def __init__(
        self,
        *,
        provider: Literal["openai", "responses", "anthropic"],
        base_query: ChatMLQuery,
        llm: Any,
        request: Request,
        conversation_messages: List[BaseMessage],
        mcp_servers: List[MCPServerConfig],
        formatter_kwargs: Optional[Dict[str, Any]] = None,
        max_iterations: int = 8,
        completion_callback: Optional[Callable[[Any], Awaitable[None]]] = None,
    ):
        self.provider = provider
        self.base_query = base_query
        self.llm = llm
        self.request = request
        self.working_messages = _clone_messages(conversation_messages)
        self.mcp_servers = mcp_servers
        self.formatter_kwargs = formatter_kwargs
        self.max_iterations = max_iterations
        self.completion_callback = completion_callback

        self.runtime: Optional[MCPRuntime] = None
        self.discovered_tools: Dict[str, List[MCPResolvedTool]] = {}
        self.resolved_by_synthetic_name: Dict[str, tuple[MCPServerConfig, MCPResolvedTool]] = {}
        self.merged_tools: List[ToolSpec] = []
        self.call_traces: List[MCPCallTrace] = []
        self.stream_formatter: Optional[Any] = None
        self.pending_stream_events = deque()

        self.stream_queue: Optional[GenQueueDynamic] = None
        self.current_query_for_turn: Optional[ChatMLQuery] = None
        self.current_shadow_task: Optional[asyncio.Task] = None
        self.current_stop_event: Optional[asyncio.Event] = None
        self.current_turn_has_suppressed_mcp = False
        self.turn_count = 0

    async def initialize(self) -> None:
        (
            self.runtime,
            self.discovered_tools,
            self.resolved_by_synthetic_name,
            self.merged_tools,
        ) = await _prepare_mcp_tools(
            base_query=self.base_query,
            mcp_servers=self.mcp_servers,
        )

    async def start(self, stream_queue: GenQueueDynamic) -> None:
        self.stream_queue = stream_queue
        await self.initialize()
        await self._start_next_turn()

    def attach_formatter(self, formatter: Any) -> None:
        self.stream_formatter = formatter
        if self.provider == "responses" and isinstance(formatter, ResponsesFormatter):
            self.pending_stream_events.extend(
                formatter.append_output_items(
                    _build_responses_mcp_list_tools_items(self.discovered_tools)
                )
            )

    def drain_stream_events(self) -> List[dict]:
        events: List[dict] = []
        while self.pending_stream_events:
            events.append(self.pending_stream_events.popleft())
        return events

    async def _start_next_turn(self) -> None:
        if self.turn_count >= self.max_iterations:
            raise HTTPException(status_code=400, detail="MCP tool loop exceeded maximum iterations")

        self.turn_count += 1
        self.current_turn_has_suppressed_mcp = False
        self.current_query_for_turn = _build_query_for_turn(
            base_query=self.base_query,
            working_messages=self.working_messages,
            merged_tools=self.merged_tools,
        )

        shadow_queue = GenQueueDynamic()
        self.current_stop_event = asyncio.Event()
        self.current_shadow_task = asyncio.create_task(
            chat_completion_response(
                query=self.current_query_for_turn,
                gen_queue=shadow_queue,
                model_name=self.llm.model_name,
                request=self.request,
                tag_definitions=self.llm.prompt_eng.tag_definitions,
                provider=self.provider,
                formatter_kwargs=self.formatter_kwargs,
            )
        )

        assert self.stream_queue is not None
        self.stream_queue.swap(GenQueue())

        asyncio.create_task(
            self.llm.chat(
                query=self.current_query_for_turn,
                prompt_eng=self.llm.prompt_eng,
                gen_queue=[self.stream_queue, shadow_queue],
                request=self.request,
                stop_event=self.current_stop_event,
            )
        )

    def request_stop_current_turn(self) -> None:
        if self.current_stop_event is not None:
            self.current_stop_event.set()

    async def intercept_tool_calls(self, tool_calls: Any) -> bool:
        if not isinstance(tool_calls, list):
            return False

        normalized_tool_calls: List[ToolCall] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                normalized_tool_calls.append(tool_call)
                continue

            if not isinstance(tool_call, dict):
                return False

            normalized_tool_calls.append(
                ToolCall(
                    id=tool_call.get("id") or "",
                    function=FunctionCall(
                        name=(tool_call.get("function") or {}).get("name", ""),
                        arguments=(tool_call.get("function") or {}).get("arguments", "{}"),
                    ),
                    type=tool_call.get("type") or "function",
                    index=tool_call.get("index"),
                )
            )

        mcp_tool_calls = _partition_mcp_tool_calls(normalized_tool_calls, self.resolved_by_synthetic_name)
        if not mcp_tool_calls:
            return False

        self.current_turn_has_suppressed_mcp = True
        return True

    async def handle_turn_end(self) -> bool:
        if not self.current_turn_has_suppressed_mcp:
            return False

        if self.current_shadow_task is None or self.current_query_for_turn is None or self.runtime is None:
            return False

        response_obj = await self.current_shadow_task
        assistant_messages = provider_response_to_assistant_messages(self.provider, response_obj)
        if not assistant_messages:
            return False

        self.working_messages.extend(_clone_messages(assistant_messages))
        latest_assistant = assistant_messages[-1]
        mcp_tool_calls = _partition_mcp_tool_calls(
            latest_assistant.tool_calls or [],
            self.resolved_by_synthetic_name,
        )
        if not mcp_tool_calls:
            return False

        executed_traces = await self._execute_mcp_tool_calls_with_stream_events(mcp_tool_calls)

        for trace in executed_traces:
            self.call_traces.append(trace)
            tool_content = trace.output_text
            if trace.error and not tool_content:
                tool_content = json.dumps(trace.error.model_dump(mode="json"), ensure_ascii=False)

            self.working_messages.append(
                BaseMessage(
                    role="tool",
                    tool_call_id=trace.call_id,
                    content=tool_content or "",
                )
            )

        await self._start_next_turn()
        return True

    async def _execute_mcp_tool_calls_with_stream_events(
        self,
        tool_calls: List[ToolCall],
    ) -> List[MCPCallTrace]:
        if self.runtime is None or self.current_query_for_turn is None:
            return []

        call_order = [tool_call.id for tool_call in tool_calls]
        traces_by_call_id: Dict[str, MCPCallTrace] = {}

        async def _run_single(tool_call: ToolCall) -> MCPCallTrace:
            server, resolved_tool = self.resolved_by_synthetic_name[tool_call.function.name]
            arguments_json = tool_call.function.arguments or "{}"
            arguments = json.loads(arguments_json)

            self._queue_mcp_call_started(
                call_id=tool_call.id,
                tool_name=resolved_tool.tool_name,
                server_name=server.name,
                arguments_json=arguments_json,
                arguments=arguments,
            )

            trace = await self.runtime.call_tool(
                server=server,
                tool=resolved_tool,
                arguments=arguments,
                call_id=tool_call.id,
            )
            self._queue_mcp_call_completed(trace)
            return trace

        if self.current_query_for_turn.parallel_tool_calls and len(tool_calls) > 1:
            tasks = [asyncio.create_task(_run_single(tool_call)) for tool_call in tool_calls]
            for completed_task in asyncio.as_completed(tasks):
                trace = await completed_task
                traces_by_call_id[trace.call_id] = trace
        else:
            for tool_call in tool_calls:
                trace = await _run_single(tool_call)
                traces_by_call_id[trace.call_id] = trace

        return [traces_by_call_id[call_id] for call_id in call_order if call_id in traces_by_call_id]

    def _queue_mcp_call_started(
        self,
        *,
        call_id: str,
        tool_name: str,
        server_name: str,
        arguments_json: str,
        arguments: Dict[str, Any],
    ) -> None:
        if self.provider == "responses" and isinstance(self.stream_formatter, ResponsesFormatter):
            self.pending_stream_events.extend(
                self.stream_formatter.append_mcp_call_started(
                    call_id=call_id,
                    arguments=arguments_json,
                    name=tool_name,
                    server_label=server_name,
                )
            )
            return

        if self.provider == "anthropic" and isinstance(self.stream_formatter, AnthropicFormatter):
            self.pending_stream_events.extend(
                self.stream_formatter.append_mcp_tool_use_block(
                    call_id=call_id,
                    name=tool_name,
                    server_name=server_name,
                    arguments=arguments,
                )
            )

    def _queue_mcp_call_completed(self, trace: MCPCallTrace) -> None:
        if self.provider == "responses" and isinstance(self.stream_formatter, ResponsesFormatter):
            self.pending_stream_events.extend(
                self.stream_formatter.append_mcp_call_completed(
                    call_id=trace.call_id,
                    output=trace.output_text,
                    error=trace.error.model_dump(mode="json") if trace.error else None,
                )
            )
            return

        if self.provider == "anthropic" and isinstance(self.stream_formatter, AnthropicFormatter):
            result_text = trace.output_text or (trace.error.message if trace.error else "")
            self.pending_stream_events.extend(
                self.stream_formatter.append_mcp_tool_result_block(
                    tool_use_id=trace.call_id,
                    output_text=result_text,
                    is_error=trace.error is not None,
                )
            )

    async def completion_callback_with_traces(self, response_obj: Any) -> None:
        response_obj = prepend_mcp_traces_to_response(
            self.provider,
            response_obj,
            self.discovered_tools,
            self.call_traces,
        )

        if self.completion_callback is not None:
            await self.completion_callback(response_obj)


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

    runtime, discovered_tools, resolved_by_synthetic_name, merged_tools = await _prepare_mcp_tools(
        base_query=base_query,
        mcp_servers=mcp_servers,
    )

    working_messages = _clone_messages(conversation_messages)
    call_traces: List[MCPCallTrace] = []

    for _ in range(max_iterations):
        query_for_turn = _build_query_for_turn(
            base_query=base_query,
            working_messages=working_messages,
            merged_tools=merged_tools,
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

        mcp_tool_calls = _partition_mcp_tool_calls(tool_calls, resolved_by_synthetic_name)
        if not mcp_tool_calls:
            return MCPLoopResult(
                prepend_mcp_traces_to_response(provider, response_obj, discovered_tools, call_traces),
                working_messages,
            )

        executed_traces = await _execute_mcp_tool_calls(
            runtime=runtime,
            query_for_turn=query_for_turn,
            resolved_by_synthetic_name=resolved_by_synthetic_name,
            tool_calls=mcp_tool_calls,
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
