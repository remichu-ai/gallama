import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from pydantic.json import pydantic_encoder

import uuid

from ..data_classes.data_class import (
    ChatCompletionResponse,
    StreamChoice,
    ChoiceDelta,
    UsageResponse,
    Choice,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicUsage,
    AnthropicToolUseBlock,
    AnthropicThinkingBlock,
    OpenAIStopReason,
    AnthropicStopReason
)


@dataclass
class ParsedContentBlock:
    """Provider-agnostic intermediate representation of a parsed content block.

    Produced by the shared parsing logic and consumed by each formatter's
    non_stream_response to build its native response format.
    """
    api_tag: str                # e.g. "content", "tool_calls", "reasoning"
    role: str                   # e.g. "assistant"
    content: Any                # processed text (str) or parsed tool list (list)
    allowed_roles: Set[str] = field(default_factory=lambda: {"assistant"})


class BaseAPIFormatter:
    def __init__(self, model_name: str, unique_id: Optional[str] = None):
        self.model_name = model_name
        self._unique_id = unique_id if unique_id else self.create_unique_id()
        self.created = int(time.time())
        self._current_block_index = 0
        self._block_active = False

    def stream_start(self) -> List[dict]:
        return []

    def open_block(self, api_tag: str) -> List[dict]:
        """Open a new content block if one is not already active. Returns events to yield."""
        events = []
        if not self._block_active:
            if b_start := self.block_start(self._current_block_index, api_tag):
                events.extend(b_start)
            self._block_active = True
        return events

    def close_block(self) -> List[dict]:
        """Close the current content block if one is active. Returns events to yield."""
        events = []
        if self._block_active:
            if b_stop := self.block_stop(self._current_block_index):
                events.extend(b_stop)
            self._block_active = False
            self._current_block_index += 1
        return events

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        return []

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        """Format a stream chunk using the internally tracked block index."""
        return []

    def block_stop(self, index: int) -> List[dict]:
        return []

    def stream_stop(self, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
                    total_tokens: Optional[int] = None, finish_reason: str = "stop") -> List[dict]:
        return []

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def current_block_index(self):
        return self._current_block_index

    def create_unique_id(self):
        raise NotImplementedError

    def non_stream_response(self, parsed_blocks: List['ParsedContentBlock'], input_tokens: int, output_tokens: int,
                            total_tokens: int, finish_reason: str = "stop") -> Any:
        raise NotImplementedError


class OpenAIFormatter(BaseAPIFormatter):
    def create_unique_id(self):
        return "cmpl-" + str(uuid.uuid4().hex)

    def stream_start(self) -> List[dict]:
        # OpenAI does not require a message start event
        return []

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        # OpenAI does not use explicit block start events
        return []

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        # Handle if text is a parsed tool call list from your post_processor
        if api_tag == "tool_calls" and isinstance(text, list):
            choice_delta_object = {
                api_tag: text,
                "role": role if role else "assistant"
            }
        else:
            choice_delta_object = {
                api_tag: str(text) if text is not None else "",
                "role": role if role else "assistant"
            }

        chunk_data = ChatCompletionResponse(
            id=self.unique_id,
            model=self.model_name,
            object="chat.completion.chunk",
            created=self.created,
            choices=[StreamChoice(index=0, delta=ChoiceDelta(**choice_delta_object))]
        )

        return [{
            "data": json.dumps(chunk_data.model_dump(exclude_unset=True), default=pydantic_encoder, ensure_ascii=False)
        }]

    def block_stop(self, index: int) -> List[dict]:
        # OpenAI does not use explicit block stop events
        return []

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: OpenAIStopReason = "stop"
    ) -> List[dict]:
        events = []

        # Emit the final chunk with finish_reason
        finish_data = ChatCompletionResponse(
            id=self.unique_id,
            model=self.model_name,
            object="chat.completion.chunk",
            created=self.created,
            choices=[StreamChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=finish_reason
            )],
        )

        events.append({"data": json.dumps(
            finish_data.model_dump(exclude_unset=True),
            default=pydantic_encoder,
            ensure_ascii=False
        )})

        # Optional usage chunk
        if input_tokens:
            usage_data = ChatCompletionResponse(
                id=self.unique_id,
                model=self.model_name,
                object="chat.completion.chunk",
                choices=[],
                usage=UsageResponse(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=total_tokens,
                ),
            )
            events.append({"data": json.dumps(usage_data.model_dump(exclude_unset=True))})

        events.append({"data": "[DONE]"})
        return events

    def non_stream_response(
        self,
        parsed_blocks: List['ParsedContentBlock'],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "stop"
    ):
        choices = []
        if parsed_blocks:
            return_obj_dict = {"role": parsed_blocks[0].role}

            for block in parsed_blocks:
                if block.api_tag in return_obj_dict:
                    existing = return_obj_dict[block.api_tag]
                    # Both lists (e.g. multiple tool_calls) — extend
                    if isinstance(existing, list) and isinstance(block.content, list):
                        return_obj_dict[block.api_tag] = existing + block.content
                    # Both strings (e.g. multiple content fragments) — concatenate
                    elif isinstance(existing, str) and isinstance(block.content, str):
                        return_obj_dict[block.api_tag] = existing + block.content
                    # Incompatible types — skip or overwrite as you see fit
                else:
                    return_obj_dict[block.api_tag] = block.content

            choices.append(Choice(**{
                "index": 0,
                "message": return_obj_dict,
                "finish_reason": finish_reason,
            }))

        return ChatCompletionResponse(
            id=self.unique_id,
            model=self.model_name,
            choices=choices,
            usage=UsageResponse(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
        )


class AnthropicFormatter(BaseAPIFormatter):
    def __init__(self, model_name: str, unique_id: Optional[str] = None):
        super().__init__(model_name, unique_id)
        self._current_api_tag = "content"

    def create_unique_id(self):
        return "msg_" + str(uuid.uuid4().hex)

    def stream_start(self) -> List[dict]:
        data = {
            "type": "message_start",
            "message": {
                "id": self.unique_id,
                "type": "message",
                "role": "assistant",
                "model": self.model_name,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }
        return [{"event": "message_start", "data": json.dumps(data)}]

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        self._current_api_tag = api_tag

        if api_tag == "tool_calls":
            # Suppress default block start. Return an empty list.
            return []

        block_type = "text" if api_tag == "content" else api_tag
        if api_tag == "reasoning":
            data = {"type": "content_block_start", "index": self._current_block_index,
                    "content_block": {"type": "thinking", "thinking": ""}}
        else:
            data = {"type": "content_block_start", "index": self._current_block_index,
                    "content_block": {"type": block_type, "text": ""}}

        return [{"event": "content_block_start", "data": json.dumps(data)}]

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        self._current_api_tag = api_tag

        if api_tag == "tool_calls":
            events = []
            parsed_tools = text if isinstance(text, list) else []

            for tool_call in parsed_tools:
                func_data = tool_call.get("function", {})
                tool_name = func_data.get("name", "unknown_tool")
                arguments = func_data.get("arguments", "{}")
                tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"

                # 1. Start block
                events.append({
                    "event": "content_block_start",
                    "data": json.dumps({
                        "type": "content_block_start",
                        "index": self._current_block_index,
                        "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name}
                    })
                })

                # 2. Delta block
                events.append({
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": self._current_block_index,
                        "delta": {"type": "input_json_delta", "partial_json": str(arguments)}
                    })
                })

                # 3. Stop block
                events.append({
                    "event": "content_block_stop",
                    "data": json.dumps({
                        "type": "content_block_stop",
                        "index": self._current_block_index
                    })
                })

                self._current_block_index += 1

            return events

        # Standard chunks
        if api_tag == "reasoning":
            delta_obj = {"type": "thinking_delta", "thinking": str(text) if text is not None else ""}
        else:
            delta_obj = {"type": "text_delta", "text": str(text) if text is not None else ""}

        data = {"type": "content_block_delta", "index": self._current_block_index, "delta": delta_obj}
        return [{"event": "content_block_delta", "data": json.dumps(data)}]

    def block_stop(self, index: int) -> List[dict]:
        if self._current_api_tag == "tool_calls":
            # Suppress default block stop for tools. Return empty list.
            return []

        data = {"type": "content_block_stop", "index": self._current_block_index}
        # NOTE: do NOT increment here — close_block() in the base class handles incrementing
        return [{"event": "content_block_stop", "data": json.dumps(data)}]

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: AnthropicStopReason = "end_turn"
    ) -> List[dict]:
        # Allow tool_calls detection from _current_api_tag to override,
        # since GenStats won't know about tool use
        if self._current_api_tag == "tool_calls":
            stop_reason = "tool_use"
        else:
            # finish_reason is already in Anthropic format from the caller
            stop_reason = finish_reason

        delta_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens or 0}
        }
        return [
            {"event": "message_delta", "data": json.dumps(delta_data)},
            {"event": "message_stop", "data": json.dumps({"type": "message_stop"})}
        ]

    def non_stream_response(
        self,
        parsed_blocks: List['ParsedContentBlock'],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "end_turn"
    ):
        content_blocks = []
        for block in parsed_blocks:
            if block.api_tag == "reasoning":
                content_blocks.append(
                    AnthropicThinkingBlock(type="thinking", thinking=block.content)
                )
            elif block.api_tag == "content":
                content_blocks.append(
                    AnthropicTextBlock(type="text", text=block.content)
                )
            elif block.api_tag == "tool_calls":
                finish_reason = "tool_use"  # overwrite to tool_use
                tool_list = block.content if isinstance(block.content, list) else []
                for tc in tool_list:
                    func_data = tc.get("function", {})
                    arguments = func_data.get("arguments", "{}")
                    content_blocks.append(
                        AnthropicToolUseBlock(
                            id=tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                            name=func_data.get("name", "unknown_tool"),
                            input=json.loads(arguments) if isinstance(arguments, str) else arguments,
                        )
                    )

        return AnthropicMessagesResponse(
            id=self.unique_id,
            model=self.model_name,
            content=content_blocks,
            stop_reason=finish_reason,
            usage=AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )