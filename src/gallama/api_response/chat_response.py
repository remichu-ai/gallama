from fastapi import Request

from .api_formatter import OpenAIFormatter, AnthropicFormatter, ResponsesFormatter, ParsedContentBlock
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

from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Literal, Optional, Union
from ..utils.request_disconnect import is_request_disconnected
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
    provider: Literal["openai", "anthropic", "responses"],
    tag_definitions: List[TagDefinition] = None,
    formatter_kwargs: Optional[Dict[str, Any]] = None,
    completion_callback: Optional[Callable[[Any], Awaitable[None]]] = None,
    tool_calls_interceptor: Optional[Callable[[Any], Awaitable[bool]]] = None,
    turn_end_interceptor: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[dict]:
    formatter_kwargs = formatter_kwargs or {}
    if provider == "openai":
        formatter = OpenAIFormatter(model_name=model_name, **formatter_kwargs)
    elif provider == "anthropic":
        formatter = AnthropicFormatter(model_name=model_name, **formatter_kwargs)
    else:
        formatter = ResponsesFormatter(model_name=model_name, **formatter_kwargs)

    text_tag = TagDefinition(tag_type="text", api_tag="content")

    if tag_definitions is not None:
        stream_parser = StreamParserByTag(tag_definitions=tag_definitions, default_tag_type=text_tag)
    else:
        stream_parser = DummyParser()

    full_response = ""
    eos = False
    gen_stats = None
    state = {}
    last_stream_event_at = time.monotonic()
    suppressed_tool_calls = False

    current_tag = text_tag
    current_text = ""

    def make_stream_parser():
        if tag_definitions is not None:
            return StreamParserByTag(tag_definitions=tag_definitions, default_tag_type=text_tag)
        return DummyParser()

    async def emit_events(events: List[dict]):
        nonlocal last_stream_event_at
        for event in events:
            last_stream_event_at = time.monotonic()
            yield event

    def process_chunk(tag: TagDefinition, text: str):
        try:
            extra_args = {"model_name": model_name, "unique_id": formatter.unique_id, "state": state}
            _processed_text = tag.post_processor(text, extra_args)
            _api_tag = tag.api_tag
            _role = tag.role
        except Exception as e:
            logger.error(f"Error in post-processor: {e}")
            logger.info("Fall back to text tag")
            _processed_text, _api_tag, _role = text, text_tag.api_tag, text_tag.role

        return _processed_text, _api_tag, _role

    async def maybe_format_chunk(tag: TagDefinition, text: str):
        nonlocal suppressed_tool_calls

        processed_text, api_tag, role = process_chunk(tag, text)

        if api_tag == "tool_calls" and tool_calls_interceptor is not None:
            if await tool_calls_interceptor(processed_text):
                suppressed_tool_calls = True
                return []

        try:
            return formatter.stream_chunk(api_tag=api_tag, text=processed_text, role=role)
        except Exception as e:
            logger.error(f"Error in stream_chunk formatting: {e}")
            raise e

    def reset_turn_state():
        nonlocal full_response
        nonlocal eos
        nonlocal gen_stats
        nonlocal state
        nonlocal current_tag
        nonlocal current_text
        nonlocal suppressed_tool_calls
        nonlocal stream_parser

        full_response = ""
        eos = False
        gen_stats = None
        state = {}
        current_tag = text_tag
        current_text = ""
        suppressed_tool_calls = False
        stream_parser = make_stream_parser()

    # Emit starting event(s)
    async for event in emit_events(formatter.stream_start()):
        yield event

    while not eos:
        accumulated_text = ""
        queue_was_empty = False
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
            queue_was_empty = True
            now = time.monotonic()
            if formatter.ping_interval_s is not None and now - last_stream_event_at >= formatter.ping_interval_s:
                async for event in emit_events(formatter.stream_ping()):
                    yield event
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
                            async for event in emit_events(formatter.open_block(current_tag.api_tag)):
                                yield event

                            # 2. Yield chunks
                            if chunk := await maybe_format_chunk(current_tag, current_text):
                                async for event in emit_events(chunk):
                                    yield event

                            current_text = ""

                        # 3. Close old block via formatter
                        async for event in emit_events(formatter.close_block()):
                            yield event

                        current_tag = _tag

                    if _tag.wait_till_complete:
                        current_text += _text_chunk
                    else:
                        async for event in emit_events(formatter.open_block(_tag.api_tag)):
                            yield event

                        if chunk := await maybe_format_chunk(_tag, _text_chunk):
                            async for event in emit_events(chunk):
                                yield event
        elif queue_was_empty and formatter.ping_interval_s is not None:
            # Keep the SSE connection active during long prefill or tool-thinking gaps.
            pass

        if eos:
            if current_text:
                async for event in emit_events(formatter.open_block(current_tag.api_tag)):
                    yield event

                if chunk := await maybe_format_chunk(current_tag, current_text):
                    async for event in emit_events(chunk):
                        yield event

            if suppressed_tool_calls and turn_end_interceptor is not None:
                if await turn_end_interceptor():
                    reset_turn_state()
                    continue

            # Close final block via formatter
            async for event in emit_events(formatter.close_block()):
                yield event

            logger.info(f"full_response: {full_response}")

            # Safe stop handling
            # Determine finish reason from gen_stats; formatter will override for tool_calls
            if provider == "openai":
                base_finish_reason = gen_stats.get_openai_stop_reason() if gen_stats else "stop"
            else:
                base_finish_reason = gen_stats.stop_reason if gen_stats else "end_turn"

            include_usage = provider != "openai" or (
                query.stream_options and getattr(query.stream_options, "include_usage", True)
            )
            if gen_stats and include_usage:
                stream_stop_events = formatter.stream_stop(
                    input_tokens=gen_stats.input_tokens_count,
                    output_tokens=gen_stats.output_tokens_count,
                    total_tokens=gen_stats.total_tokens_count,
                    finish_reason=base_finish_reason,
                )
            else:
                stream_stop_events = formatter.stream_stop(finish_reason=base_finish_reason)

            async for event in emit_events(stream_stop_events):
                yield event

            if completion_callback is not None:
                try:
                    response_obj = None
                    final_response = getattr(formatter, "final_stream_response", None)
                    if callable(final_response):
                        response_obj = final_response(
                            input_tokens=gen_stats.input_tokens_count if gen_stats else None,
                            output_tokens=gen_stats.output_tokens_count if gen_stats else None,
                            total_tokens=gen_stats.total_tokens_count if gen_stats else None,
                        )
                    if response_obj is not None:
                        await completion_callback(response_obj)
                except Exception as e:
                    logger.error(f"Error in response completion callback: {e}")

            if gen_stats:
                logger.info(format_generation_stats_log(model_name, gen_stats))


async def chat_completion_response(
    query: ChatMLQuery,
    gen_queue: GenQueueDynamic,
    model_name: str,
    request: Request,
    provider: Literal["openai", "anthropic", "responses"],
    tag_definitions: List[TagDefinition] = None,
    formatter_kwargs: Optional[Dict[str, Any]] = None,
    completion_callback: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> Union[ChatCompletionResponse, AnthropicMessagesResponse, Any]:
    formatter_kwargs = formatter_kwargs or {}
    if provider == "openai":
        formatter = OpenAIFormatter(model_name=model_name, **formatter_kwargs)
    elif provider == "anthropic":
        formatter = AnthropicFormatter(model_name=model_name, **formatter_kwargs)
    else:
        formatter = ResponsesFormatter(model_name=model_name, **formatter_kwargs)

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
                logger.debug(f"Stream starts with {item.gen_type}")
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
            try:
                processed_content = _tag.post_processor(_text_chunk)
                processed_api_tag = _tag.api_tag
                processed_role = _tag.role
                processed_allowed_roles = _tag.allowed_roles
            except Exception as e:
                logger.error(f"Error in post-processor: {e}")
                logger.info("Fall back to text tag")
                processed_content = _text_chunk
                processed_api_tag = text_tag.api_tag
                processed_role = text_tag.role
                processed_allowed_roles = text_tag.allowed_roles

            parsed_blocks.append(ParsedContentBlock(
                api_tag=processed_api_tag,
                role=processed_role,
                content=processed_content,
                allowed_roles=processed_allowed_roles,
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

    if completion_callback is not None:
        try:
            await completion_callback(response_obj)
        except Exception as e:
            logger.error(f"Error in response completion callback: {e}")

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
                if await asyncio.wait_for(is_request_disconnected(request), timeout=0.1):
                    logger.info("Client disconnected, stopping stream")
                    break
            except asyncio.TimeoutError:
                pass
    if not eos:
        logger.info("Stream ended before receiving GenEnd")
