from fastapi import Request

from .api_formatter import OpenAIFormatter, AnthropicFormatter, ResponsesFormatter, ParsedContentBlock
from .stream_parser_v2 import StreamParserByTag, DummyParser
from ..data_classes.data_class import (
    ChatMLQuery,
    ChatCompletionResponse,
    ChatMessage,
    ParsedToolCall,
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
from ..logger.logger import basic_log_extra
from pydantic.json import pydantic_encoder
import time
import json
import asyncio
import uuid
import logging


def _materialize_tool_call(call: Any, index: int) -> Dict[str, Any]:
    if hasattr(call, "model_dump"):
        call = call.model_dump(exclude_none=True)

    if isinstance(call, dict) and "function" in call:
        function_payload = call.get("function", {})
        raw_arguments = function_payload.get("arguments", "{}")
        arguments_json = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments)

        return ChoiceDeltaToolCall(
            index=call.get("index", index),
            id=call.get("id") or get_response_tool_uid(),
            function=ChoiceDeltaToolCallFunction(
                name=function_payload.get("name"),
                arguments=arguments_json,
            ),
            type=call.get("type", "function"),
        ).model_dump(exclude_unset=True)

    if isinstance(call, dict):
        parsed_call = ParsedToolCall(
            name=call.get("name", ""),
            arguments=call.get("arguments", {}),
        )
    else:
        parsed_call = ParsedToolCall.model_validate(call)

    return ChoiceDeltaToolCall(
        index=index,
        id=get_response_tool_uid(),
        function=ChoiceDeltaToolCallFunction(
            name=parsed_call.name,
            arguments=json.dumps(parsed_call.arguments),
        ),
        type="function",
    ).model_dump(exclude_unset=True)


def _materialize_tool_calls(calls: Any, start_index: int) -> tuple[Any, int]:
    if not isinstance(calls, list):
        return calls, start_index

    materialized_calls = []
    next_index = start_index
    for call in calls:
        materialized_calls.append(_materialize_tool_call(call, next_index))
        next_index += 1

    return materialized_calls, next_index


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


def format_stream_start_log(gen_type: Union[TagDefinition, str]) -> str:
    if logger.isEnabledFor(logging.DEBUG):
        return f"Stream starts with {gen_type}"

    tag_type = getattr(gen_type, "tag_type", gen_type)
    return f"Stream starts with {tag_type}"


def _resolve_gen_queue(
    gen_queue: GenQueue | GenQueueDynamic | List[GenQueue | GenQueueDynamic],
) -> GenQueue | GenQueueDynamic:
    if isinstance(gen_queue, list):
        return gen_queue[0]
    return gen_queue


def _drain_available_queue_items(
    gen_queue: GenQueue | GenQueueDynamic,
    *,
    first_item: Any,
) -> List[Any]:
    items = [first_item]
    while True:
        try:
            items.append(gen_queue.get_nowait())
        except asyncio.QueueEmpty:
            return items


async def _wait_for_queue_item(
    gen_queue: GenQueue | GenQueueDynamic,
    *,
    timeout: Optional[float] = None,
) -> Any:
    try:
        return gen_queue.get_nowait()
    except asyncio.QueueEmpty:
        pass

    if timeout is None:
        return await gen_queue.get()

    return await asyncio.wait_for(gen_queue.get(), timeout=timeout)


def _remaining_ping_timeout(
    *,
    last_event_at: float,
    ping_interval_s: Optional[float],
) -> Optional[float]:
    if ping_interval_s is None:
        return None

    return max(0.0, ping_interval_s - (time.monotonic() - last_event_at))


async def get_response_from_queue(
    gen_queue: GenQueue | GenQueueDynamic | List[GenQueue|GenQueueDynamic],
    request: Request = None,
):
    """ function to get the text generated in queue to be used for other part of the library"""
    response = ""

    gen_queue_to_use = _resolve_gen_queue(gen_queue)

    eos = False
    gen_stats = None
    while not eos:
        result = await _wait_for_queue_item(gen_queue_to_use)
        for result in _drain_available_queue_items(gen_queue_to_use, first_item=result):
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
    formatter_ready_callback: Optional[Callable[[Any], None]] = None,
    extra_events_getter: Optional[Callable[[], List[dict]]] = None,
    generation_stop_callback: Optional[Callable[[], None]] = None,
) -> AsyncIterator[dict]:
    formatter_kwargs = formatter_kwargs or {}
    if provider == "openai":
        formatter = OpenAIFormatter(model_name=model_name, **formatter_kwargs)
    elif provider == "anthropic":
        formatter = AnthropicFormatter(model_name=model_name, **formatter_kwargs)
    else:
        formatter = ResponsesFormatter(model_name=model_name, **formatter_kwargs)

    if formatter_ready_callback is not None:
        formatter_ready_callback(formatter)

    text_tag = TagDefinition(tag_type="text", api_tag="content")

    if tag_definitions is not None:
        stream_parser = StreamParserByTag(tag_definitions=tag_definitions, default_tag_type=text_tag)
    else:
        stream_parser = DummyParser()

    full_response = ""
    eos = False
    gen_stats = None
    next_tool_call_index = 0
    last_stream_event_at = time.monotonic()
    suppressed_tool_calls = False
    generation_stop_requested = False

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

    async def emit_extra_events():
        if extra_events_getter is None:
            return

        events = extra_events_getter() or []
        if not events:
            return

        async for event in emit_events(events):
            yield event

    def process_chunk(tag: TagDefinition, text: str):
        nonlocal next_tool_call_index

        try:
            extra_args = {"model_name": model_name, "unique_id": formatter.unique_id}
            _processed_text = tag.post_processor(text, extra_args)
            _api_tag = tag.api_tag
            _role = tag.role
            if _api_tag == "tool_calls":
                _processed_text, next_tool_call_index = _materialize_tool_calls(
                    _processed_text,
                    next_tool_call_index,
                )
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
        nonlocal next_tool_call_index
        nonlocal current_tag
        nonlocal current_text
        nonlocal suppressed_tool_calls
        nonlocal generation_stop_requested
        nonlocal stream_parser

        full_response = ""
        eos = False
        gen_stats = None
        next_tool_call_index = 0
        current_tag = text_tag
        current_text = ""
        suppressed_tool_calls = False
        generation_stop_requested = False
        stream_parser = make_stream_parser()

    # Emit starting event(s)
    async for event in emit_events(formatter.stream_start()):
        yield event
    async for event in emit_extra_events():
        yield event

    while not eos:
        accumulated_text = ""
        try:
            item = await _wait_for_queue_item(
                gen_queue,
                timeout=_remaining_ping_timeout(
                    last_event_at=last_stream_event_at,
                    ping_interval_s=formatter.ping_interval_s,
                ),
            )
            for item in _drain_available_queue_items(gen_queue, first_item=item):
                if isinstance(item, GenText):
                    accumulated_text += item.content
                elif isinstance(item, GenEnd):
                    eos = True
                    break
                elif isinstance(item, GenStart):
                    logger.info(format_stream_start_log(item.gen_type))
                    if current_tag != item.gen_type:
                        current_tag = item.gen_type
                        try:
                            stream_parser.push_tag_context(item.gen_type)
                        except Exception as e:
                            logger.error(e)
                elif isinstance(item, GenerationStats):
                    gen_stats = item
        except asyncio.QueueEmpty:
            pass
        except asyncio.TimeoutError:
            if formatter.ping_interval_s is not None:
                async for event in emit_events(formatter.stream_ping()):
                    yield event
            continue

        if accumulated_text or eos:
            full_response += accumulated_text
            parsed_text = stream_parser.process_stream(accumulated_text)
            if (
                stream_parser.generation_should_stop
                and generation_stop_callback is not None
                and not generation_stop_requested
            ):
                logger.info("Stopping generation after invalid continuation following a restricted tag")
                generation_stop_callback()
                generation_stop_requested = True

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

        if eos:
            if current_text:
                async for event in emit_events(formatter.open_block(current_tag.api_tag)):
                    yield event

                if chunk := await maybe_format_chunk(current_tag, current_text):
                    async for event in emit_events(chunk):
                        yield event

            if suppressed_tool_calls and turn_end_interceptor is not None:
                turn_end_task = asyncio.create_task(turn_end_interceptor())
                while True:
                    if turn_end_task.done():
                        break

                    async for event in emit_extra_events():
                        yield event

                    wait_timeout = _remaining_ping_timeout(
                        last_event_at=last_stream_event_at,
                        ping_interval_s=formatter.ping_interval_s,
                    )
                    try:
                        await asyncio.wait_for(asyncio.shield(turn_end_task), timeout=wait_timeout)
                    except asyncio.TimeoutError:
                        async for event in emit_events(formatter.stream_ping()):
                            yield event

                if await turn_end_task:
                    async for event in emit_extra_events():
                        yield event
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
                stream_stop_kwargs = {
                    "input_tokens": gen_stats.input_tokens_count,
                    "output_tokens": gen_stats.output_tokens_count,
                    "total_tokens": gen_stats.total_tokens_count,
                    "finish_reason": base_finish_reason,
                }
                if provider == "anthropic":
                    stream_stop_kwargs["stop_sequence"] = gen_stats.stop_sequence
                stream_stop_events = formatter.stream_stop(**stream_stop_kwargs)
            else:
                stream_stop_kwargs = {"finish_reason": base_finish_reason}
                if provider == "anthropic" and gen_stats:
                    stream_stop_kwargs["stop_sequence"] = gen_stats.stop_sequence
                stream_stop_events = formatter.stream_stop(**stream_stop_kwargs)

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
                logger.info(format_generation_stats_log(model_name, gen_stats), extra=basic_log_extra())


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
    next_tool_call_index = 0
    while not eos:
        item = await _wait_for_queue_item(gen_queue)
        for item in _drain_available_queue_items(gen_queue, first_item=item):
            if isinstance(item, GenText):
                response += item.content
            elif isinstance(item, GenerationStats):
                gen_stats = item        # Not applicable for completion endpoint
            elif isinstance(item, GenStart):
                logger.debug(format_stream_start_log(item.gen_type))
                if text_tag != item.gen_type:
                    initial_tag = item.gen_type

            elif isinstance(item, GenEnd):
                eos = True
                gen_queue.task_done()
                logger.info("----------------------LLM Response---------------\n" + response.strip())


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
                if processed_api_tag == "tool_calls":
                    processed_content, next_tool_call_index = _materialize_tool_calls(
                        processed_content,
                        next_tool_call_index,
                    )
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

    non_stream_kwargs = {
        "parsed_blocks": parsed_blocks,
        "input_tokens": gen_stats.input_tokens_count if gen_stats else 0,
        "output_tokens": gen_stats.output_tokens_count if gen_stats else 0,
        "total_tokens": gen_stats.total_tokens_count if gen_stats else 0,
        "finish_reason": finish_reason,
    }
    if provider == "anthropic" and gen_stats:
        non_stream_kwargs["stop_sequence"] = gen_stats.stop_sequence

    response_obj = formatter.non_stream_response(**non_stream_kwargs)

    assert response_obj is not None
    # logger.info(f"full_response: {response}")
    if gen_stats:
        logger.info(format_generation_stats_log(model_name, gen_stats), extra=basic_log_extra())

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
            result = await _wait_for_queue_item(gen_queue, timeout=0.1)
            for result in _drain_available_queue_items(gen_queue, first_item=result):
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
        except asyncio.TimeoutError:
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
                logger.info(format_generation_stats_log(model_name, gen_stats), extra=basic_log_extra())
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
