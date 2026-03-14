from fastapi import Request

from .api_formatter import OpenAIFormatter, AnthropicFormatter, ParsedContentBlock
from .stream_parser_v2 import StreamParserByTag, DummyParser
from ..data_classes.data_class import (
    ChatMLQuery,
    ChatCompletionResponse,
    ChatMessage,
    UsageResponse,
    OneTool,
    ToolCallResponse,
    StreamChoice,
    CompletionResponse,
    CompletionStreamResponse,
    CompletionChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TextTag,
    TagDefinition,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
)
from ..data_classes.generation_data_class import (
    GenerationStats,
    GenQueue,
    GenText,
    GenEnd,
    GenStart,
    GenQueueDynamic
)

from typing import AsyncIterator, List, Dict, Literal, Union
from ..utils.utils import get_response_uid, get_response_tool_uid
from ..logger import logger
from pydantic.json import pydantic_encoder
import time
import json
import asyncio
import uuid


def format_generation_stats_log(model_name: str, gen_stats: GenerationStats) -> str:
    parts = [
        f"{model_name} | generation {gen_stats.generation_speed:.1f} tok/s",
        f"prefill {gen_stats.prefill_speed:.1f} tok/s",
        f"input {gen_stats.input_tokens_count}",
        f"output {gen_stats.output_tokens_count}",
        f"total {gen_stats.total_tokens_count}",
        f"ttft {gen_stats.time_to_first_token:.2f}s",
        f"gen {gen_stats.time_generate:.2f}s",
        f"total_time {gen_stats.total_time:.2f}s",
        f"stop {gen_stats.stop_reason}",
    ]

    if gen_stats.cached_tokens:
        parts.append(f"cached_tokens {gen_stats.cached_tokens}")
    if gen_stats.cached_pages:
        parts.append(f"cached_pages {gen_stats.cached_pages}")

    return " | ".join(parts)


async def get_response_from_queue(
    gen_queue: GenQueue | GenQueueDynamic | List[GenQueue|GenQueueDynamic],
    request: Request = None,
):
    """ function to get the text generated in queue to be used for other part of the library"""
    response = ""

    # if it is a list, assume it is the first element
    gen_queue_to_use = gen_queue
    if isinstance(gen_queue, list):
        gen_queue_to_use = gen_queue[0]

    eos = False
    gen_stats = None
    while not eos:
        try:
            result = gen_queue_to_use.get_nowait()

            if isinstance(result, GenText):
                response += result.content
            elif isinstance(result, GenerationStats):
                gen_stats = result
            elif isinstance(result, GenStart):
                pass
            elif isinstance(result, GenEnd):
                eos = True
                gen_queue_to_use.task_done()
                logger.info("----------------------LLM Response---------------\n" + response.strip())

        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)    # short sleep before trying again

    return response, gen_stats


async def chat_completion_response_stream(
    query: ChatMLQuery,
    gen_queue: GenQueueDynamic,
    model_name: str,
    request: Request,
    provider: Literal["openai", "anthropic"],
    tag_definitions: List[TagDefinition] = None
) -> AsyncIterator[dict]:

    formatter_class = OpenAIFormatter if provider == "openai" else AnthropicFormatter
    formatter = formatter_class(model_name=model_name)

    text_tag = TagDefinition(tag_type="text", api_tag="content")

    if tag_definitions is not None:
        stream_parser = StreamParserByTag(tag_definitions=tag_definitions, default_tag_type=text_tag)
    else:
        stream_parser = DummyParser()

    full_response = ""
    eos = False
    gen_stats = None
    state = {}

    current_tag = text_tag
    current_text = ""

    def process_and_format_chunk(tag: TagDefinition, text: str):
        try:
            extra_args = {"model_name": model_name, "unique_id": formatter.unique_id, "state": state}
            _processed_text = tag.post_processor(text, extra_args)
            _api_tag = tag.api_tag
            _role = tag.role
        except Exception as e:
            logger.error(f"Error in post-processor: {e}")
            logger.info("Fall back to text tag")
            _processed_text, _api_tag, _role = text, text_tag.api_tag, text_tag.role

        try:
            return formatter.stream_chunk(api_tag=_api_tag, text=_processed_text, role=_role)
        except Exception as e:
            logger.error(f"Error in stream_chunk formatting: {e}")
            raise e

    # Emit starting event(s)
    for event in formatter.stream_start():
        yield event

    while not eos:
        accumulated_text = ""
        try:
            while True:
                item = gen_queue.get_nowait()
                if isinstance(item, GenText):
                    accumulated_text += item.content
                elif isinstance(item, GenEnd):
                    eos = True
                    break
                elif isinstance(item, GenStart):
                    logger.info(f"Stream starts with {item.gen_type}")
                    if current_tag != item.gen_type:
                        current_tag = item.gen_type
                        try:
                            stream_parser.push_tag_context(item.gen_type)
                        except Exception as e:
                            logger.error(e)
                elif isinstance(item, GenerationStats):
                    gen_stats = item
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)

        if accumulated_text or eos:
            full_response += accumulated_text
            parsed_text = stream_parser.process_stream(accumulated_text)

            if eos:
                parsed_text.extend(stream_parser.flush())

            if parsed_text:
                for _tag, _text_chunk in parsed_text:
                    if current_tag != _tag:
                        logger.info(f"new tag: {_tag}")
                        if current_text:
                            # 1. Open block via formatter
                            for e in formatter.open_block(current_tag.api_tag):
                                yield e

                            # 2. Yield chunks
                            if chunk := process_and_format_chunk(current_tag, current_text):
                                for c in chunk:
                                    yield c

                            current_text = ""

                        # 3. Close old block via formatter
                        for e in formatter.close_block():
                            yield e

                        current_tag = _tag

                    if _tag.wait_till_complete:
                        current_text += _text_chunk
                    else:
                        for e in formatter.open_block(_tag.api_tag):
                            yield e

                        if chunk := process_and_format_chunk(_tag, _text_chunk):
                            for c in chunk:
                                yield c

        if eos:
            if current_text:
                for e in formatter.open_block(current_tag.api_tag):
                    yield e

                if chunk := process_and_format_chunk(current_tag, current_text):
                    for c in chunk:
                        yield c

            # Close final block via formatter
            for e in formatter.close_block():
                yield e

            logger.info(f"full_response: {full_response}")

            # Safe stop handling
            # Determine finish reason from gen_stats; formatter will override for tool_calls
            if provider == "openai":
                base_finish_reason = gen_stats.get_openai_stop_reason() if gen_stats else "stop"
            else:
                base_finish_reason = gen_stats.stop_reason if gen_stats else "end_turn"

            if gen_stats and query.stream_options and getattr(query.stream_options, "include_usage", True):
                stream_stop_events = formatter.stream_stop(
                    input_tokens=gen_stats.input_tokens_count,
                    output_tokens=gen_stats.output_tokens_count,
                    total_tokens=gen_stats.total_tokens_count,
                    finish_reason=base_finish_reason,
                )
            else:
                stream_stop_events = formatter.stream_stop(finish_reason=base_finish_reason)

            for event in stream_stop_events:
                yield event

            if gen_stats:
                logger.info(format_generation_stats_log(model_name, gen_stats))


async def chat_completion_response(
    query: ChatMLQuery,
    gen_queue: GenQueueDynamic,
    model_name: str,
    request: Request,
    provider: Literal["openai", "anthropic"],
    tag_definitions: List[TagDefinition] = None,
) -> ChatCompletionResponse:

    formatter_class = OpenAIFormatter if provider == "openai" else AnthropicFormatter
    formatter = formatter_class(model_name=model_name)

    # create streaming pro
    text_tag = TagDefinition(
        tag_type="text",
        api_tag="content",
    )

    if tag_definitions is not None:
        stream_parser = StreamParserByTag(
            tag_definitions=tag_definitions,
            default_tag_type=text_tag
        )
    else:
        stream_parser = DummyParser()

    response = ""
    initial_tag = None
    gen_stats = None
    eos = False
    while not eos:
        try:

            item = gen_queue.get_nowait()
            if isinstance(item, GenText):
                response += item.content
            elif isinstance(item, GenerationStats):
                gen_stats = item        # Not applicable for completion endpoint
            elif isinstance(item, GenStart):
                logger.info(f"Stream starts with {item.gen_type}")
                if text_tag != item.gen_type:
                    initial_tag = item.gen_type

            elif isinstance(item, GenEnd):
                eos = True
                gen_queue.task_done()
                logger.info("----------------------LLM Response---------------\n" + response.strip())

        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)    # short sleep before trying again


    response = response.strip()
    parsed_text = stream_parser.parse_full_text(response, initial_tag=initial_tag)

    # Build provider-agnostic parsed blocks
    parsed_blocks = []
    if parsed_text:
        for _tag, _text_chunk in parsed_text:
            parsed_blocks.append(ParsedContentBlock(
                api_tag=_tag.api_tag,
                role=_tag.role,
                content=_tag.post_processor(_text_chunk),
                allowed_roles=_tag.allowed_roles,
            ))

    # Determine finish reason from content
    # Base finish reason from gen_stats, mapped per provider
    if provider == "openai":
        finish_reason = gen_stats.get_openai_stop_reason() if gen_stats else "stop"
    else:
        finish_reason = gen_stats.stop_reason if gen_stats else "end_turn"

    # Override for tool use (gen_stats won't know about tool calls)
    for block in parsed_blocks:
        if block.api_tag == "tool_calls":
            finish_reason = "tool_calls" if provider == "openai" else "tool_use"
            break

    response_obj = formatter.non_stream_response(
        parsed_blocks=parsed_blocks,
        input_tokens=gen_stats.input_tokens_count,
        output_tokens=gen_stats.output_tokens_count,
        total_tokens=gen_stats.total_tokens_count,
        finish_reason=finish_reason,
    )

    assert response_obj is not None
    # logger.info(f"full_response: {response}")
    if gen_stats:
        logger.info(format_generation_stats_log(model_name, gen_stats))

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
                    finish_reason=gen_stats.get_openai_stop_reason() if gen_stats else "stop"
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
                logger.info(format_generation_stats_log(model_name, gen_stats))
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
