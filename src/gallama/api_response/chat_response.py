from fastapi import Request
from gallama.data_classes.data_class import (
    ChatMLQuery,
    ChatCompletionResponse,
    Choice,
    ChatMessage,
    UsageResponse,
    OneTool,
    ToolCallResponse,
    StreamChoice,
    GenerationStats,
    GenQueue,
    GenText,
    GenEnd,
    GenStart,
    CompletionResponse,
    CompletionStreamResponse,
    CompletionChoice,
    TextTag,
)
from .stream_parser import StreamParser
from typing import AsyncIterator
from gallama.utils.utils import get_response_uid, get_response_tool_uid
from gallama.logger.logger import logger
from pydantic.json import pydantic_encoder
import time
import json
import asyncio


async def get_response_from_queue(
    gen_queue: GenQueue,
    request: Request = None,
):
    """ function to get the text generated in queue to be used for other part of the library"""
    response = ""
    # global result_queue
    # completed_event = asyncio.Event()

    eos = False
    genStats = None
    while not eos:
        try:
            # if request is not None:
            #     if await request.is_disconnected():
            #         logger.info("Request disconnected, stopping queue processing")
            #         break

            result = gen_queue.get_nowait()

            if isinstance(result, GenText):
                response += result.content
            elif isinstance(result, GenerationStats):
                genStats = result
            elif isinstance(result, GenStart):
                pass
            elif isinstance(result, GenEnd):
                eos = True
                gen_queue.task_done()
                logger.info("----------------------LLM Response---------------\n" + response.strip())

        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)    # short sleep before trying again

    return response, genStats


async def chat_completion_response_stream(
    query: ChatMLQuery,
    gen_queue: GenQueue,
    model_name: str,
    request: Request,
) -> AsyncIterator[dict]:
    unique_id = get_response_uid()
    created = int(time.time())
    full_response = ""
    eos = False
    gen_type = "text"  # Default generation type
    gen_stats = None

    # last_log_time = time.time()
    # log_interval = 1  # Log every 5 seconds

    while not eos:
        # Logging to troubleshoot if queue build up
        # current_time = time.time()
        #
        # # Log queue size every 5 seconds
        # if current_time - last_log_time >= log_interval:
        #     queue_size = gen_queue.qsize()
        #     logger.info(f"Queue size: {queue_size}")
        #     last_log_time = current_time

        # if await request.is_disconnected():
        #     logger.info("Request disconnected, stopping queue processing")
        #     break

        accumulated_text = ""
        accumulated_thinking = ""

        try:
            # Collect all available items from the queue
            while True:
                item = gen_queue.get_nowait()
                if isinstance(item, GenText) and (item.text_type=="text"):
                    accumulated_text += item.content
                elif isinstance(item, GenText) and (item.text_type=="tool"):
                    accumulated_text += item.content
                elif isinstance(item, GenText) and item.text_type=="thinking":
                    accumulated_thinking += item.content
                elif isinstance(item, GenEnd):
                    eos = True
                    break
                elif isinstance(item, GenStart):
                    gen_type = item.gen_type
                elif isinstance(item, GenerationStats):
                    gen_stats = item
        except asyncio.QueueEmpty:
            pass

        if accumulated_text or accumulated_thinking:
            full_response += accumulated_thinking + accumulated_text

            # TODO current naive implementation with assumption that thinking will always come first instead of the actual order that text stream in

            if gen_type == "text" or gen_type == "thinking":

                # if the output is returned together, then concat into one text
                if query.return_thinking is True and accumulated_thinking:
                    if accumulated_text:
                        accumulated_text = "\n" + accumulated_text
                    # assumption: thinking come first
                    accumulated_text = accumulated_thinking + accumulated_text

                # yield thinking response first
                if accumulated_thinking and query.return_thinking=="separate":
                    chunk_data = ChatCompletionResponse(
                        unique_id=unique_id,
                        model=model_name,
                        object="chat.completion.chunk",
                        created=created,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChatMessage(
                                    role="assistant",
                                    content="",
                                    thinking=accumulated_thinking,
                                ),
                            )
                        ],
                    )
                    yield {"data": json.dumps(chunk_data.model_dump(exclude_unset=True), default=pydantic_encoder, ensure_ascii=False)}

                # yield text response
                if accumulated_text:
                    chunk_data = ChatCompletionResponse(
                        unique_id=unique_id,
                        model=model_name,
                        object="chat.completion.chunk",
                        created=created,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=ChatMessage(
                                    role="assistant",
                                    content=accumulated_text,
                                ),
                            )
                        ],
                    )
                    yield {"data": json.dumps(chunk_data.model_dump(exclude_unset=True), default=pydantic_encoder, ensure_ascii=False)}
            elif gen_type == "tool":
                # Accumulate tool usage data
                # Note: This assumes that tool data is complete in a single chunk
                # If tool data can span multiple chunks, you'll need to implement a more sophisticated accumulation strategy
                tool_response = json.loads(accumulated_text)
                tools_list = []
                for index, tool in enumerate(tool_response.get('functions_calling', [])):
                    tool_id = get_response_tool_uid()
                    tools_list.append(
                        ToolCallResponse(
                            id=tool_id,
                            index=index,
                            function=OneTool(
                                name=tool['name'],
                                arguments=json.dumps(tool['arguments']),
                            )
                        )
                    )
                chunk_data = ChatCompletionResponse(
                    unique_id=unique_id,
                    model=model_name,
                    object="chat.completion.chunk",
                    created=created,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=ChatMessage(
                                role="assistant",
                                tool_calls=tools_list,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                yield {
                    "event": "message",
                    "data": json.dumps(chunk_data.model_dump(exclude_unset=True), default=pydantic_encoder, ensure_ascii=False)
                }

        if eos:
            # Log the full response at the end
            logger.info(f"----------------------LLM Response---------------\n{full_response.strip()}")

            # Include generation stats if available and requested
            if gen_stats and query.stream_options and query.stream_options.include_usage:
                usage_data = ChatCompletionResponse(
                    unique_id=unique_id,
                    model=model_name,
                    object="chat.completion.chunk",
                    choices=[],
                    usage=UsageResponse(
                        prompt_tokens=gen_stats.input_tokens_count,
                        completion_tokens=gen_stats.output_tokens_count,
                        total_tokens=gen_stats.total_tokens_count,
                    ),
                )
                yield {"data": json.dumps(usage_data.model_dump(exclude_unset=True))}

            if gen_stats:
                logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")

            # Send the ending DONE message
            yield {"data": "[DONE]"}
        else:
            await asyncio.sleep(0.1)  # Short sleep before next iteration if not at end of stream


async def chat_completion_response(
    query: ChatMLQuery,
    gen_queue: GenQueue,
    # response: str,
    # gen_stats: GenerationStats,
    model_name: str,
    request: Request,
    # mode: Literal["text", "tool"] = "text"
) -> ChatCompletionResponse:

    response = ""
    response_thinking = ""
    response_all = ""
    # global result_queue
    # completed_event = asyncio.Event()
    gen_type = "text"
    gen_stats = None
    eos = False
    while not eos:
        try:
            # if await request.is_disconnected():
            #     logger.info("Request disconnected, stopping queue processing")
            #     break

            result = gen_queue.get_nowait()
            if isinstance(result, GenText) and result.text_type=="text":
                response += result.content
                response_all += result.content
            elif isinstance(result, GenText) and result.text_type=="tool":
                response += result.content
                response_all += result.content
            elif isinstance(result, GenText) and result.text_type=="thinking":
                response_thinking += result.content
                response_all += result.content
            elif isinstance(result, GenerationStats):
                gen_stats = result        # Not applicable for completion endpoint
            elif isinstance(result, GenStart):
                gen_type = result
                gen_type = result.gen_type      # get the gen_type e.g. text, tool, thinking
            elif isinstance(result, GenEnd):
                eos = True
                gen_queue.task_done()
                logger.info("----------------------LLM Response---------------\n" + response_all.strip())

        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)    # short sleep before trying again


    unique_id = get_response_uid()
    response = response.strip()

    if gen_type == "text" or gen_type=="thinking":

        # whether to return separate or together
        if query.return_thinking is False:
            response_thinking = ""
        elif query.return_thinking is True:
            response = response_thinking + "\n" + response
            # response_thinking = ""     # still return the response_thinking for user to be able to segregate
        elif query.return_thinking == "separate":
            pass

        response_obj = ChatCompletionResponse(
            unique_id=unique_id,
            model=model_name,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response,
                        thinking=response_thinking,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageResponse(
                prompt_tokens=gen_stats.input_tokens_count,
                completion_tokens=gen_stats.output_tokens_count,
                total_tokens=gen_stats.total_tokens_count,
            ),
        )
    elif gen_type == "tool":
        try:
            response_dict = json.loads(response)
        except:
            # since out put is not tool, return it as text instead #TODO find better solution
            response_obj = ChatCompletionResponse(
                unique_id=unique_id,
                model=model_name,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=UsageResponse(
                    prompt_tokens=gen_stats.input_tokens_count,
                    completion_tokens=gen_stats.output_tokens_count,
                    total_tokens=gen_stats.total_tokens_count,
                ),
            )

        # successfully parse JSON, convert the tool used into response format
        tools_list = []  # the list of tool to call
        for index, tool in enumerate(response_dict['functions_calling']):
            tool_id = get_response_tool_uid()
            tools_list.append(
                ToolCallResponse(
                    id=tool_id,
                    index=index,
                    function=OneTool(
                        name=tool['name'],
                        arguments=json.dumps(tool['arguments']),
                    )
                )
            )

        response_obj = ChatCompletionResponse(
            model=model_name,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=tools_list,
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=UsageResponse(
                prompt_tokens=gen_stats.input_tokens_count,
                completion_tokens=gen_stats.output_tokens_count,
                total_tokens=gen_stats.total_tokens_count,
            ),
        )

    assert response_obj is not None
    logger.debug("----------------------LLM API Response---------------\n" + json.dumps(response_obj.model_dump(), indent=2))
    logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")

    return response_obj


async def completion_response(
    gen_queue: GenQueue,
    model_name: str,
    request: Request,
) -> CompletionResponse:
    response, gen_stats = await get_response_from_queue(gen_queue, request=request)

    if response:
        completion_response = CompletionResponse(
            model=model_name,
            choices=[
                CompletionChoice(
                    text=response.strip(),
                    index=0,
                    logprobs=None,
                    finish_reason="stop"  # You may want to determine this based on actual finish reason
                )
            ],
            usage=UsageResponse(
                prompt_tokens=gen_stats.input_tokens_count if gen_stats else 0,
                completion_tokens=gen_stats.output_tokens_count if gen_stats else 0,
                total_tokens=gen_stats.total_tokens_count if gen_stats else 0
            )
        )

        # Use model_dump() and json.dumps() instead of json() method
        logger.info(
            f"----------------------LLM API Response---------------\n{json.dumps(completion_response.model_dump(), indent=2)}")
        return completion_response


async def completion_response_stream(
    request: Request,
    gen_queue: GenQueue,
    model_name: str
) -> AsyncIterator:
    unique_id = get_response_uid()
    full_response = ""
    eos = False
    gen_stats = None
    while not eos:
        # if await request.is_disconnected():
        #     logger.info("Request disconnected, stopping queue processing")
        #     break

        accumulated_text = ""
        try:
            while True:
                result = gen_queue.get_nowait()
                if isinstance(result, GenText):
                    accumulated_text += result.content
                elif isinstance(result, GenEnd):
                    eos = True
                    gen_queue.task_done()
                    break
                elif isinstance(result, GenStart):
                    pass
                elif isinstance(result, GenerationStats):
                    gen_stats = result
        except asyncio.QueueEmpty:
            pass
        if accumulated_text:
            full_response += accumulated_text
            chunk_data = CompletionStreamResponse(
                id=unique_id,
                object="text_completion",
                created=int(time.time()),
                model=model_name,
                system_fingerprint="fp_44709d6fcb",
                choices=[
                    CompletionChoice(
                        text=accumulated_text,
                        index=0,
                        logprobs=None,
                        finish_reason=None
                    )
                ]
            )
            json_data = json.dumps(chunk_data.model_dump())
            if json_data.strip():
                logger.debug(f"Yielding: {json_data!r}")
                yield json_data
        if eos:
            logger.info(f"----------------------LLM Response---------------\n{full_response.strip()}")
            if gen_stats is not None:
                logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")
            yield "[DONE]"
            break
        else:
            try:
                if await asyncio.wait_for(request.is_disconnected(), timeout=0.1):
                    logger.info("Client disconnected, stopping stream")
                    break
            except asyncio.TimeoutError:
                pass
    if not eos:
        logger.info("Stream ended before receiving GenEnd")


async def chat_completion_response_artifact_stream(
    query: ChatMLQuery,
    gen_queue: GenQueue,
    model_name: str,
    request: Request,
) -> AsyncIterator[dict]:
    unique_id = get_response_uid()
    created = int(time.time())
    full_response = ""
    response_thinking = ""
    eos = False
    gen_type = "text"  # Default generation type
    gen_stats = None

    # last_log_time = time.time()
    # log_interval = 1  # Log every 5 seconds

    artifact_parser = StreamParser()
    malformed_data = False
    MALFORMED_CHECK_LENGTH = 70     # the length limit of text so that the content_type block appear
    content_type = None    # either text or code

    while not eos:
        accumulated_text = ""
        accumulated_thinking = ""

        # if await request.is_disconnected():
        #     logger.info("Request disconnected, stopping queue processing")
        #     break

        try:
            # Collect all available items from the queue
            while True:
                item = gen_queue.get_nowait()
                if isinstance(item, GenText) and item.text_type=="text":
                    accumulated_text += item.content
                if isinstance(item, GenText) and item.text_type=="tool":
                    accumulated_text += item.content
                elif isinstance(item, GenText) and item.text_type=="thinking":
                    accumulated_thinking += item.content
                elif isinstance(item, GenEnd):
                    eos = True
                    break
                elif isinstance(item, GenStart):
                    gen_type = item.gen_type
                elif isinstance(item, GenerationStats):
                    gen_stats = item
        except asyncio.QueueEmpty:
            pass

        if accumulated_thinking and query.return_thinking is not False:
            full_response += accumulated_thinking
            chunk_data = ChatCompletionResponse(
                unique_id=unique_id,
                model=model_name,
                object="chat.completion.chunk",
                created=created,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=ChatMessage(
                            role="assistant",
                            content="",
                            thinking=accumulated_thinking,
                        ),
                    )
                ],
            )
            yield {"data": json.dumps(chunk_data.model_dump(exclude_unset=True), default=pydantic_encoder,
                                      ensure_ascii=False)}

        if accumulated_text:
            full_response += accumulated_text

            if gen_type == "text":
                parsed_chunks = artifact_parser.process_stream(accumulated_text)

                for chunk_type, chunk_content in parsed_chunks:
                    if chunk_content:
                        chunk_data = ChatCompletionResponse(
                            unique_id=unique_id,
                            model=model_name,
                            object="chat.completion.chunk",
                            created=created,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=ChatMessage(
                                        role="assistant",
                                        content=chunk_content,
                                        artifact_meta=chunk_type.model_dump()

                                    ),
                                ),
                            ],
                        )
                        yield {"data": json.dumps(chunk_data.model_dump(exclude_unset=True))}

            elif gen_type == "tool":
                # artifact do not affect tool usage
                # Accumulate tool usage data
                # Note: This assumes that tool data is complete in a single chunk
                # If tool data can span multiple chunks, you'll need to implement a more sophisticated accumulation strategy
                tool_response = json.loads(accumulated_text)
                tools_list = []
                for index, tool in enumerate(tool_response.get('functions_calling', [])):
                    tool_id = get_response_tool_uid()
                    tools_list.append(
                        ToolCallResponse(
                            id=tool_id,
                            index=index,
                            function=OneTool(
                                name=tool['name'],
                                arguments=json.dumps(tool['arguments']),
                            )
                        )
                    )
                chunk_data = ChatCompletionResponse(
                    unique_id=unique_id,
                    model=model_name,
                    object="chat.completion.chunk",
                    created=created,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=ChatMessage(
                                role="assistant",
                                tool_calls=tools_list,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                yield {"data": json.dumps(chunk_data.dict(exclude_unset=True))}

        if eos:
            # Log the full response at the end
            logger.info(f"----------------------LLM Response---------------\n{full_response.strip()}")

            # Include generation stats if available and requested
            if gen_stats and query.stream_options and query.stream_options.include_usage:
                usage_data = ChatCompletionResponse(
                    unique_id=unique_id,
                    model=model_name,
                    object="chat.completion.chunk",
                    choices=[],
                    usage=UsageResponse(
                        prompt_tokens=gen_stats.input_tokens_count,
                        completion_tokens=gen_stats.output_tokens_count,
                        total_tokens=gen_stats.total_tokens_count,
                    ),
                )
                yield {"data": json.dumps(usage_data.model_dump(exclude_unset=True))}

            if gen_stats:
                logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")

            # Send the ending DONE message
            yield {"data": "[DONE]"}
        else:
            await asyncio.sleep(0.1)  # Short sleep before next iteration if not at end of stream


async def chat_completion_response_artifact(
    query: ChatMLQuery,
    gen_queue: GenQueue,
    model_name: str,
    request: Request,
) -> ChatCompletionResponse:
    response = ""
    response_thinking = ""
    response_all = ""

    response_obj = None
    gen_type = GenStart(gen_type="text")
    gen_stats = None
    eos = False

    while not eos:
        try:
            # if await request.is_disconnected():
            #     logger.info("Request disconnected, stopping queue processing")
            #     break

            result = gen_queue.get_nowait()
            if isinstance(result, GenText) and result.text_type=="text":
                response += result.content
                response_all += result.content
            elif isinstance(result, GenText) and result.text_type=="tool":
                response += result.content
                response_all += result.content
            elif isinstance(result, GenText) and result.text_type=="thinking":
                response_thinking += result.content
                response_all += result.content
            elif isinstance(result, GenerationStats):
                gen_stats = result
            elif isinstance(result, GenStart):
                gen_type = result
            elif isinstance(result, GenEnd):
                eos = True
                gen_queue.task_done()
                logger.info("----------------------LLM Response---------------\n" + response_all.strip())
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)

    unique_id = get_response_uid()
    response = response.strip()

    if gen_type.gen_type == "text" or gen_type=="thinking":
        choices = []

        # return thinking first
        if response_thinking and query.return_thinking is not False:    # currently always return as separate
            choices.append(
                Choice(
                    index=0,    # only 1 thinking per response for now
                    message=ChatMessage(
                        role="assistant",
                        content='',
                        thinking=response_thinking,
                    ),
                    finish_reason="stop"
                )
            )

        parser = StreamParser()
        parsed_chunks = parser.parse_full_response(response)

        if parsed_chunks:
            # If parsing was successful, create a structured response

            for idx, (chunk_type, chunk_content) in enumerate(parsed_chunks):
                choices.append(
                    Choice(
                        index=idx,
                        message=ChatMessage(
                            role="assistant",
                            content=chunk_content,
                            artifact_meta=chunk_type.model_dump(),
                        ),
                        finish_reason="stop"
                    )
                )
        else:
            # If parsing failed, treat the entire response as a single text chunk
            choices = [
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response,
                        artifact_meta=TextTag().model_dump()
                    ),
                    finish_reason="stop"
                )
            ]

        response_obj = ChatCompletionResponse(
            unique_id=unique_id,
            model=model_name,
            choices=choices,
            usage=UsageResponse(
                prompt_tokens=gen_stats.input_tokens_count if gen_stats else 0,
                completion_tokens=gen_stats.output_tokens_count if gen_stats else 0,
                total_tokens=gen_stats.total_tokens_count if gen_stats else 0,
            ),
        )

    elif gen_type.gen_type == "tool":
        try:
            response_dict = json.loads(response)
        except:
            # since output is not tool, return it as text instead #TODO find better solution
            response_obj = ChatCompletionResponse(
                unique_id=unique_id,
                model=model_name,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response,
                            artifact_meta=TextTag().model_dump()
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=UsageResponse(
                    prompt_tokens=gen_stats.input_tokens_count,
                    completion_tokens=gen_stats.output_tokens_count,
                    total_tokens=gen_stats.total_tokens_count,
                ),
            )
        else:
            # successfully parse JSON, convert the tool used into response format
            tools_list = []  # the list of tool to call
            for index, tool in enumerate(response_dict['functions_calling']):
                tool_id = get_response_tool_uid()
                tools_list.append(
                    ToolCallResponse(
                        id=tool_id,
                        index=index,
                        function=OneTool(
                            name=tool['name'],
                            arguments=json.dumps(tool['arguments']),
                        )
                    )
                )

            response_obj = ChatCompletionResponse(
                model=model_name,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=None,
                            tool_calls=tools_list,
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=UsageResponse(
                    prompt_tokens=gen_stats.input_tokens_count,
                    completion_tokens=gen_stats.output_tokens_count,
                    total_tokens=gen_stats.total_tokens_count,
                ),
            )

    assert response_obj is not None
    logger.debug("----------------------LLM API Response---------------\n" + json.dumps(response_obj.model_dump(), indent=2))
    logger.info(f"{model_name} | LLM speed {gen_stats.generation_speed:.1f}/s tokens")

    return response_obj
