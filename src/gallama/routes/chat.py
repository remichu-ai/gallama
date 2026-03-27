from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from ..api_response.chat_response import (
    chat_completion_response_stream,
    chat_completion_response,
    completion_response,
    completion_response_stream,
)
from ..dependencies import get_model_manager, get_response_store
from ..logger import logger
from ..remote_mcp.orchestrator import MCPStreamController, prepend_mcp_traces_to_response, run_mcp_completion_loop
from ..request_validation import validate_api_request
from ..data_classes import (
    BaseMessage,
    ChatMLQuery,
    GenerateQuery,
    GenQueue,
    GenQueueDynamic,
    AnthropicMessagesRequest,
    ResponsesCreateRequest,
    ResponsesCreateResponse,
    response_output_to_assistant_messages,
)
from ..response_store import ResponseStoreRecord
from typing import Literal

import asyncio

# https://platform.openai.com/docs/api-reference/chat/create

router = APIRouter(prefix="/v1", tags=["chat"])


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
                    tool_calls_interceptor=controller.intercept_tool_calls,
                    turn_end_interceptor=controller.handle_turn_end,
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
        # llm = model_manager.llm_dict.get(query.model)
        # if not llm:
        #     llm = model_manager.llm_dict[list(model_manager.llm_dict.keys())[0]]

        # start the generation task
        asyncio.create_task(
            llm.chat(
                query=query,
                prompt_eng=llm.prompt_eng,
                gen_queue=gen_queue,
                request=request,
            )
        )

        # send the response to client
        if query.stream:
            # EventSourceResponse take iterator so need to handle at here
            return EventSourceResponse(
                chat_completion_response_stream(
                    query=query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider=provider
                ))

        else:
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


@router.post("/responses")
async def responses(request: Request, query: ResponsesCreateRequest):
    model_manager = get_model_manager()
    response_store = get_response_store()
    gen_queue = GenQueueDynamic()
    should_store = query.store or query.previous_response_id is not None
    response_request = query.model_copy(update={"store": should_store})

    previous_messages: list[BaseMessage] = []
    if query.previous_response_id:
        previous_record = await response_store.get(query.previous_response_id)
        if previous_record is None:
            raise HTTPException(status_code=404, detail=f"Response '{query.previous_response_id}' not found")
        previous_messages = previous_record.conversation_messages

    current_messages = query.to_input_messages(include_instructions=not previous_messages)
    effective_messages = (
        _merge_instructions_into_history(previous_messages, query.instructions) + current_messages
        if previous_messages
        else current_messages
    )

    try:
        raw_chat_query = response_request.to_chat_ml_query(messages=effective_messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        mcp_servers = response_request.get_mcp_server_configs()

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
                    if not should_store:
                        return

                    stored_response = prepend_mcp_traces_to_response(
                        "responses",
                        response_obj,
                        controller.discovered_tools,
                        controller.call_traces,
                    )
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
                            store=should_store,
                        )
                    )

                return EventSourceResponse(
                    chat_completion_response_stream(
                        query=raw_chat_query,
                        gen_queue=gen_queue,
                        model_name=llm.model_name,
                        request=request,
                        tag_definitions=llm.prompt_eng.tag_definitions,
                        provider="responses",
                        formatter_kwargs=formatter_kwargs,
                        completion_callback=mcp_stream_completion_callback if should_store else None,
                        tool_calls_interceptor=controller.intercept_tool_calls,
                        turn_end_interceptor=controller.handle_turn_end,
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
                        store=should_store,
                    )
                )

            return result.response_obj

        chat_query = validate_api_request(raw_chat_query)
        llm = model_manager.get_model(chat_query.model, _type="llm")

        async def completion_callback(response_obj: ResponsesCreateResponse):
            if not should_store:
                return
            assistant_messages = response_output_to_assistant_messages(response_obj.output)
            conversation_messages = _clone_messages(effective_messages) + assistant_messages
            await response_store.put(
                ResponseStoreRecord(
                    response_id=response_obj.id,
                    model=response_obj.model,
                    request=response_request.model_dump(exclude_none=True, by_alias=True),
                    response=response_obj,
                    conversation_messages=conversation_messages,
                    previous_response_id=response_request.previous_response_id,
                    store=should_store,
                )
            )

        asyncio.create_task(
            llm.chat(
                query=chat_query,
                prompt_eng=llm.prompt_eng,
                gen_queue=gen_queue,
                request=request,
            )
        )

        formatter_kwargs = {"request_model": response_request}

        if chat_query.stream:
            return EventSourceResponse(
                chat_completion_response_stream(
                    query=chat_query,
                    gen_queue=gen_queue,
                    model_name=llm.model_name,
                    request=request,
                    tag_definitions=llm.prompt_eng.tag_definitions,
                    provider="responses",
                    formatter_kwargs=formatter_kwargs,
                    completion_callback=completion_callback if should_store else None,
                )
            )

        return await chat_completion_response(
            query=chat_query,
            gen_queue=gen_queue,
            model_name=llm.model_name,
            request=request,
            tag_definitions=llm.prompt_eng.tag_definitions,
            provider="responses",
            formatter_kwargs=formatter_kwargs,
            completion_callback=completion_callback if should_store else None,
        )
    except HTTPException as e:
        logger.error(e)
        return e


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

    # start the generation task
    asyncio.create_task(llm.chat_raw(
        prompt=query.prompt,
        max_tokens=query.max_tokens,
        gen_queue=gen_queue,
        request=request,
    ))

    if query.stream:
        # EventSourceResponse take iterator so need to handle iterator here
        return EventSourceResponse(completion_response_stream(request=request, gen_queue=gen_queue, model_name=llm.model_name))
    else:
        return await completion_response(gen_queue=gen_queue, model_name=llm.model_name, request=request)
