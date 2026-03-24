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
from ..data_classes import (
    BaseMessage,
    ChatMLQuery,
    ToolForce,
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


def validate_api_request(query: ChatMLQuery):
    # fixing api call parameter
    # update temperature to 0.1 because huggingface doesn't accept 0 temperature
    if query.temperature <= 0:
        query.temperature = 0.01
    # tool_choice is set to 'auto' if there is tool

    # set tool_choice if user didn't set it and ensure parameters are correct
    # if no tool, set tool_choice to none
    if query.tools:
        if query.tool_choice is None:
            if query.tools:
                query.tool_choice = "auto"
            else:
                query.tool_choice = "none"
        elif query.tool_choice == "none":
            # wipe out all the tools if any
            query.tools = None
        elif isinstance(query.tool_choice, ToolForce):
            # wipe out all the tool except the tool to be forced 'tool_choice' in api call
            force_single_tool = []
            for tool in query.tools:
                if tool.function.name == query.tool_choice.function.name:
                    force_single_tool.append(tool)

            if len(force_single_tool) != 1:
                raise Exception("tool_choice not exist in function list")
            else:
                # query.tool_choice = "required"
                query.tools = force_single_tool
    else:
        query.tool_choice = "none"

    # validate that prefix_strings and regex_prefix_pattern  or regex_pattern can not be used together
    if query.prefix_strings and (query.regex_pattern or query.regex_prefix_pattern):
        raise HTTPException(status_code=400, detail="refix_strings and regex_pattern, regex_prefix_pattern can not be used together")

    return query


@router.post("/messages")
async def anthropic_message(request: Request, message: AnthropicMessagesRequest):
    if message.strip_claude_code_billing_header:
        removed = message.remove_claude_code_billing_header_system_message()
        if removed:
            logger.info("Removed Claude Code billing header from Anthropic system prompt for prompt caching")

    return await chat_completion(
        request,
        message.get_ChatMLQuery(),
        provider="anthropic"
    )

@router.post("/chat/completions")
async def chat_completion(request: Request, query: ChatMLQuery, provider: Literal["openai", "anthropic"]="openai"):

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
        chat_query = validate_api_request(response_request.to_chat_ml_query(messages=effective_messages))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
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
