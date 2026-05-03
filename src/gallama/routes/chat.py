from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from ..api_response.chat_response import (
    chat_completion_response_stream,
    chat_completion_response,
    completion_response,
    completion_response_stream,
)
from ..dependencies import get_conversation_store, get_model_manager, get_response_store
from ..logger import logger
from ..remote_mcp.orchestrator import MCPStreamController, prepend_mcp_traces_to_response, run_mcp_completion_loop
from ..request_validation import validate_api_request
from ..data_classes import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    BaseMessage,
    ChatMLQuery,
    GenerateQuery,
    GenEnd,
    GenQueue,
    GenQueueDynamic,
    AnthropicMessagesRequest,
    ConversationCreateRequest,
    ConversationDeletedResource,
    ConversationUpdateRequest,
    normalize_input_messages,
    ResponsesCreateRequest,
    ResponsesCreateResponse,
    response_output_to_assistant_messages,
)
from ..response_store import ResponseStoreRecord
from ..server_engine.responses_ws_bridge import is_codex_responses_transport
from ..utils.utils import get_token_length
from typing import Literal

import asyncio

# https://platform.openai.com/docs/api-reference/chat/create

router = APIRouter(prefix="/v1", tags=["chat"])


def _log_generation_task_failure(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        logger.error("Background generation task failed with %s: %s", exc.status_code, exc.detail)
        return

    logger.exception("Background generation task failed")


async def _run_generation_task(coro, gen_queue: GenQueueDynamic) -> None:
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log_generation_task_failure(exc)
        try:
            gen_queue.put_nowait(GenEnd())
        except Exception:
            logger.exception("Failed to terminate generation queue after background task error")


def _start_generation_task(coro, gen_queue: GenQueueDynamic) -> asyncio.Task:
    task = asyncio.create_task(_run_generation_task(coro, gen_queue))

    def _consume_task_exception(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            logger.info("Background generation task cancelled")
        except Exception:
            # _run_generation_task already logged the failure and terminated the queue.
            pass

    task.add_done_callback(_consume_task_exception)
    return task


def _count_query_input_tokens(query: ChatMLQuery) -> int:
    model_manager = get_model_manager()

    query = validate_api_request(query)
    llm = model_manager.get_model(query.model, _type="llm")
    query = llm.validate_video_support(query)

    prompt_output = llm.prompt_eng.get_prompt(
        query,
        backend=llm.backend,
    )
    prompt = prompt_output[0] if isinstance(prompt_output, tuple) else prompt_output

    input_tokens = get_token_length(llm.tokenizer, prompt)
    llm.validate_token_length(input_tokens)
    return input_tokens


@router.post("/messages")
async def anthropic_message(request: Request, message: AnthropicMessagesRequest):
    if message.strip_claude_code_billing_header:
        removed = message.remove_claude_code_billing_header_system_message()
        if removed:
            logger.info("Removed Claude Code billing header from Anthropic system prompt for prompt caching")

    if message.get_mcp_server_configs():
        model_manager = get_model_manager()
        llm = model_manager.get_model(message.model, _type="llm")
        base_query = message.get_ChatMLQuery()
        if base_query.stream:
            gen_queue = GenQueueDynamic()
            controller = MCPStreamController(
                provider="anthropic",
                base_query=base_query,
                llm=llm,
                request=request,
                conversation_messages=base_query.messages,
                mcp_servers=message.get_mcp_server_configs(),
            )
            await controller.start(gen_queue)

            return EventSourceResponse(
                chat_completion_response_stream(
                    query=base_query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider="anthropic",
                    formatter_ready_callback=controller.attach_formatter,
                    extra_events_getter=controller.drain_stream_events,
                    tool_calls_interceptor=controller.intercept_tool_calls,
                    turn_end_interceptor=controller.handle_turn_end,
                    generation_stop_callback=controller.request_stop_current_turn,
                )
            )

        result = await run_mcp_completion_loop(
            provider="anthropic",
            base_query=base_query,
            llm=llm,
            request=request,
            conversation_messages=base_query.messages,
            mcp_servers=message.get_mcp_server_configs(),
        )
        return result.response_obj

    return await chat_completion(request, message.get_ChatMLQuery(), provider="anthropic")


@router.post("/messages/count_tokens")
async def anthropic_count_tokens(message: AnthropicCountTokensRequest) -> AnthropicCountTokensResponse:
    if message.strip_claude_code_billing_header:
        removed = message.remove_claude_code_billing_header_system_message()
        if removed:
            logger.info("Removed Claude Code billing header from Anthropic system prompt for prompt caching")

    input_tokens = _count_query_input_tokens(message.get_ChatMLQuery())
    return AnthropicCountTokensResponse(input_tokens=input_tokens)

@router.post("/chat/completions")
async def chat_completion(request: Request, query: ChatMLQuery, provider: Literal["openai", "anthropic"]="openai"):
    function_tools, mcp_servers = query.split_tools()
    if mcp_servers:
        model_manager = get_model_manager()
        llm = model_manager.get_model(query.model, _type="llm")
        base_query = query.model_copy(deep=True, update={"tools": function_tools})
        if base_query.stream:
            gen_queue = GenQueueDynamic()
            controller = MCPStreamController(
                provider=provider,
                base_query=base_query,
                llm=llm,
                request=request,
                conversation_messages=base_query.messages,
                mcp_servers=mcp_servers,
            )
            await controller.start(gen_queue)

            return EventSourceResponse(
                chat_completion_response_stream(
                    query=base_query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider=provider,
                    tool_calls_interceptor=controller.intercept_tool_calls,
                    turn_end_interceptor=controller.handle_turn_end,
                    generation_stop_callback=controller.request_stop_current_turn,
                )
            )

        result = await run_mcp_completion_loop(
            provider=provider,
            base_query=base_query,
            llm=llm,
            request=request,
            conversation_messages=base_query.messages,
            mcp_servers=mcp_servers,
        )
        return result.response_obj

    model_manager = get_model_manager()
    gen_queue = GenQueueDynamic()      # this queue will hold the result for this generation

    # validate and fix query
    query = validate_api_request(query)

    try:
        llm = model_manager.get_model(query.model, _type="llm")
        stop_event = asyncio.Event()
        # llm = model_manager.llm_dict.get(query.model)
        # if not llm:
        #     llm = model_manager.llm_dict[list(model_manager.llm_dict.keys())[0]]

        if query.stream:
            _start_generation_task(
                llm.chat(
                    query=query,
                    prompt_eng=llm.prompt_eng,
                    gen_queue=gen_queue,
                    request=request,
                    stop_event=stop_event,
                ),
                gen_queue,
            )
            # EventSourceResponse take iterator so need to handle at here
            return EventSourceResponse(
                chat_completion_response_stream(
                    query=query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider=provider,
                    generation_stop_callback=stop_event.set,
                ))

        await llm.chat(
            query=query,
            prompt_eng=llm.prompt_eng,
            gen_queue=gen_queue,
            request=request,
            stop_event=stop_event,
        )
        return await chat_completion_response(
            query=query,
            gen_queue=gen_queue,
            model_name=llm.model_name,
            request=request,
            tag_definitions=llm.prompt_eng.tag_definitions,
            provider=provider
        )
    except HTTPException as e:
        logger.error(e)
        return e


def _clone_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [message.model_copy(deep=True) for message in messages]


def _merge_instructions_into_history(
    history_messages: list[BaseMessage],
    instructions: str | None,
) -> list[BaseMessage]:
    merged_messages = _clone_messages(history_messages)
    if not instructions:
        return merged_messages

    if merged_messages and merged_messages[0].role in {"system", "developer"} and isinstance(merged_messages[0].content, str):
        base_content = merged_messages[0].content or ""
        merged_messages[0].content = f"{base_content}\n\n{instructions}" if base_content else instructions
        return merged_messages

    return [BaseMessage(role="system", content=instructions)] + merged_messages


def _missing_conversation(conversation_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")


@router.post("/responses")
async def responses(request: Request, query: ResponsesCreateRequest):
    model_manager = get_model_manager()
    conversation_store = get_conversation_store()
    response_store = get_response_store()
    gen_queue = GenQueueDynamic()
    should_store = query.store or query.previous_response_id is not None
    conversation_param = query.get_conversation_param()
    conversation_id = conversation_param.id if conversation_param else None
    response_request = query.model_copy(
        update={
            "store": should_store,
            "conversation": conversation_param,
        }
    )

    previous_messages: list[BaseMessage] = []
    if query.previous_response_id:
        previous_record = await response_store.get(query.previous_response_id)
        if previous_record is None:
            raise HTTPException(status_code=404, detail=f"Response '{query.previous_response_id}' not found")
        previous_messages = previous_record.conversation_messages
    elif conversation_id:
        conversation_record = await conversation_store.get(conversation_id)
        if conversation_record is None:
            raise _missing_conversation(conversation_id)
        previous_messages = _clone_messages(conversation_record.messages)

    ensure_user_for_codex = is_codex_responses_transport(request.headers)
    current_messages = query.to_input_messages(include_instructions=not previous_messages)
    conversation_input_messages = query.to_input_messages(include_instructions=False) if conversation_id else []
    effective_messages = (
        _merge_instructions_into_history(previous_messages, query.instructions) + current_messages
        if previous_messages
        else current_messages
    )
    effective_messages = normalize_input_messages(
        effective_messages,
        ensure_user=ensure_user_for_codex,
    )

    try:
        raw_chat_query = response_request.to_chat_ml_query(messages=effective_messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        mcp_servers = response_request.get_mcp_server_configs()
        parsed_tool_types = [
            getattr(tool, "type", type(tool).__name__)
            for tool in (response_request.tools or [])
        ]

        if parsed_tool_types:
            logger.info(
                "Responses request parsed tool types: %s",
                parsed_tool_types,
            )

        if mcp_servers:
            logger.info(
                "Responses request detected %s MCP server(s): %s",
                len(mcp_servers),
                [server.name for server in mcp_servers],
            )

        async def completion_callback(response_obj: ResponsesCreateResponse):
            assistant_messages = response_output_to_assistant_messages(response_obj.output)
            conversation_messages = _clone_messages(effective_messages) + assistant_messages

            if should_store:
                await response_store.put(
                    ResponseStoreRecord(
                        response_id=response_obj.id,
                        model=response_obj.model,
                        request=response_request.model_dump(exclude_none=True, by_alias=True),
                        response=response_obj,
                        conversation_messages=conversation_messages,
                        previous_response_id=response_request.previous_response_id,
                        conversation_id=conversation_id,
                        store=should_store,
                    )
                )

            if conversation_id:
                updated_record = await conversation_store.append_messages(
                    conversation_id,
                    conversation_input_messages + assistant_messages,
                )
                if updated_record is None:
                    logger.warning("Conversation '%s' disappeared before completion persisted", conversation_id)

        if mcp_servers:
            llm = model_manager.get_model(raw_chat_query.model, _type="llm")
            formatter_kwargs = {"request_model": response_request}

            if raw_chat_query.stream:
                controller = MCPStreamController(
                    provider="responses",
                    base_query=raw_chat_query,
                    llm=llm,
                    request=request,
                    conversation_messages=effective_messages,
                    mcp_servers=mcp_servers,
                    formatter_kwargs=formatter_kwargs,
                )
                await controller.start(gen_queue)

                async def mcp_stream_completion_callback(response_obj: ResponsesCreateResponse):
                    stored_response = response_obj
                    if should_store:
                        assistant_messages = response_output_to_assistant_messages(response_obj.output)
                        conversation_messages = _clone_messages(controller.working_messages) + assistant_messages
                        await response_store.put(
                            ResponseStoreRecord(
                                response_id=stored_response.id,
                                model=stored_response.model,
                                request=response_request.model_dump(exclude_none=True, by_alias=True),
                                response=stored_response,
                                conversation_messages=conversation_messages,
                                previous_response_id=response_request.previous_response_id,
                                conversation_id=conversation_id,
                                store=should_store,
                            )
                        )

                    if conversation_id:
                        assistant_messages = response_output_to_assistant_messages(response_obj.output)
                        updated_record = await conversation_store.append_messages(
                            conversation_id,
                            conversation_input_messages + assistant_messages,
                        )
                        if updated_record is None:
                            logger.warning("Conversation '%s' disappeared before completion persisted", conversation_id)

                return EventSourceResponse(
                    chat_completion_response_stream(
                        query=raw_chat_query,
                        gen_queue=gen_queue,
                        model_name=llm.model_name,
                        request=request,
                        tag_definitions=llm.prompt_eng.tag_definitions,
                        provider="responses",
                        formatter_kwargs=formatter_kwargs,
                        formatter_ready_callback=controller.attach_formatter,
                        completion_callback=(
                            mcp_stream_completion_callback
                            if should_store or conversation_id
                            else None
                        ),
                        extra_events_getter=controller.drain_stream_events,
                        tool_calls_interceptor=controller.intercept_tool_calls,
                        turn_end_interceptor=controller.handle_turn_end,
                        generation_stop_callback=controller.request_stop_current_turn,
                    )
                )

            result = await run_mcp_completion_loop(
                provider="responses",
                base_query=raw_chat_query,
                llm=llm,
                request=request,
                conversation_messages=effective_messages,
                mcp_servers=mcp_servers,
                formatter_kwargs=formatter_kwargs,
            )

            if should_store:
                await response_store.put(
                    ResponseStoreRecord(
                        response_id=result.response_obj.id,
                        model=result.response_obj.model,
                        request=response_request.model_dump(exclude_none=True, by_alias=True),
                        response=result.response_obj,
                        conversation_messages=_clone_messages(result.conversation_messages),
                        previous_response_id=response_request.previous_response_id,
                        conversation_id=conversation_id,
                        store=should_store,
                    )
                )

            if conversation_id:
                assistant_messages = response_output_to_assistant_messages(result.response_obj.output)
                updated_record = await conversation_store.append_messages(
                    conversation_id,
                    conversation_input_messages + assistant_messages,
                )
                if updated_record is None:
                    logger.warning("Conversation '%s' disappeared before completion persisted", conversation_id)

            return result.response_obj

        chat_query = validate_api_request(raw_chat_query)
        llm = model_manager.get_model(chat_query.model, _type="llm")
        stop_event = asyncio.Event()

        formatter_kwargs = {"request_model": response_request}

        if chat_query.stream:
            _start_generation_task(
                llm.chat(
                    query=chat_query,
                    prompt_eng=llm.prompt_eng,
                    gen_queue=gen_queue,
                    request=request,
                    stop_event=stop_event,
                ),
                gen_queue,
            )
            return EventSourceResponse(
                chat_completion_response_stream(
                    query=chat_query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider="responses",
                    formatter_kwargs=formatter_kwargs,
                    completion_callback=completion_callback if should_store or conversation_id else None,
                    generation_stop_callback=stop_event.set,
                )
            )

        await llm.chat(
            query=chat_query,
            prompt_eng=llm.prompt_eng,
            gen_queue=gen_queue,
            request=request,
            stop_event=stop_event,
        )
        return await chat_completion_response(
            query=chat_query,
            gen_queue=gen_queue,
            model_name=llm.model_name,
            request=request,
            tag_definitions=llm.prompt_eng.tag_definitions,
            provider="responses",
            formatter_kwargs=formatter_kwargs,
            completion_callback=completion_callback if should_store or conversation_id else None,
        )
    except HTTPException as e:
        logger.error(e)
        return e


@router.post("/conversations")
async def create_conversation(query: ConversationCreateRequest):
    conversation_store = get_conversation_store()
    record = await conversation_store.create(
        metadata=query.metadata,
        messages=query.to_messages(),
    )
    return record.to_resource()


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation_store = get_conversation_store()
    record = await conversation_store.get(conversation_id)
    if record is None:
        raise _missing_conversation(conversation_id)
    return record.to_resource()


@router.post("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, query: ConversationUpdateRequest):
    conversation_store = get_conversation_store()
    record = await conversation_store.update_metadata(conversation_id, query.metadata)
    if record is None:
        raise _missing_conversation(conversation_id)
    return record.to_resource()


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conversation_store = get_conversation_store()
    record = await conversation_store.delete(conversation_id)
    if record is None:
        raise _missing_conversation(conversation_id)
    return ConversationDeletedResource(id=conversation_id)


@router.get("/responses/{response_id}")
async def get_response(response_id: str):
    response_store = get_response_store()
    record = await response_store.get(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found")
    return record.response


@router.delete("/responses/{response_id}")
async def delete_response(response_id: str):
    response_store = get_response_store()
    record = await response_store.delete(response_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found")
    return {
        "id": response_id,
        "object": "response.deleted",
        "deleted": True,
    }


@router.post("/chat/generate")
@router.post("/completions")
async def generate(request: Request, query: GenerateQuery):
    model_manager = get_model_manager()
    gen_queue = GenQueueDynamic()      # this queue will hold the result for this generation
    llm = model_manager.get_model(query.model, _type="llm")
    # llm = model_manager.llm_dict[query.model]

    if query.stream:
        _start_generation_task(
            llm.chat_raw(
                prompt=query.prompt,
                max_tokens=query.max_tokens,
                gen_queue=gen_queue,
                request=request,
            ),
            gen_queue,
        )
        # EventSourceResponse take iterator so need to handle iterator here
        return EventSourceResponse(completion_response_stream(request=request, gen_queue=gen_queue, model_name=llm.model_name))

    await llm.chat_raw(
        prompt=query.prompt,
        max_tokens=query.max_tokens,
        gen_queue=gen_queue,
        request=request,
    )
    return await completion_response(gen_queue=gen_queue, model_name=llm.model_name, request=request)
