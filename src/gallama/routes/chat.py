from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from ..api_response.chat_response import (
    chat_completion_response_stream,
    chat_completion_response,
    completion_response,
    completion_response_stream,
    chat_completion_response_artifact_stream,
chat_completion_response_artifact
)
from ..dependencies import get_model_manager
from ..logger import logger
from gallama.data_classes import (
    ChatMLQuery,
    ToolForce,
    GenerateQuery,
    GenQueue,
    GenQueueDynamic
)
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
                query.tool_choice = "required"
                query.tools = force_single_tool
    else:
        query.tool_choice = "none"

    # validate that prefix_strings and regex_prefix_pattern  or regex_pattern can not be used together
    if query.prefix_strings and (query.regex_pattern or query.regex_prefix_pattern):
        raise HTTPException(status_code=400, detail="refix_strings and regex_pattern, regex_prefix_pattern can not be used together")

    return query


@router.post("/chat/completions")
async def chat_completion(request: Request, query: ChatMLQuery):

    model_manager = get_model_manager()
    gen_queue = GenQueueDynamic()      # this queue will hold the result for this generation

    # validate and fix query
    query = validate_api_request(query)

    try:
        llm = model_manager.get_model(query.model, _type="llm")
        # llm = model_manager.llm_dict.get(query.model)
        # if not llm:
        #     llm = model_manager.llm_dict[list(model_manager.llm_dict.keys())[0]]

        # log if thinking is used
        if query.thinking_template:
            logger.info(f"thinking is used with returnThinking set to {query.return_thinking}")

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
            if query.artifact == "No":     # not using artefact
                return EventSourceResponse(
                    chat_completion_response_stream(
                        query=query, gen_queue=gen_queue, model_name=llm.model_name, request=request,
                    ))
            else:
                return EventSourceResponse(
                    chat_completion_response_artifact_stream(
                        query=query, gen_queue=gen_queue, model_name=llm.model_name, request=request,
                    ))
        else:
            if query.artifact == "No":     # not using artefact
                return await chat_completion_response(query=query, gen_queue=gen_queue, model_name=llm.model_name, request=request,)
            else:
                return await chat_completion_response_artifact(query=query, gen_queue=gen_queue, model_name=llm.model_name, request=request,)
    except HTTPException as e:
        logger.error(e)
        return e


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
        return EventSourceResponse(completion_response_stream(request, gen_queue, model_name=llm.model_name, request=request,))
    else:
        return await completion_response(gen_queue, model_name=llm.model_name, request=request,)