import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic.json import pydantic_encoder

from ..data_classes.data_class import (
    AnthropicMessagesResponse,
    AnthropicStopReason,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    OpenAIStopReason,
    StreamChoice,
    UsageResponse,
)
from ..data_classes.responses_api import (
    ResponseFunctionCallItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryConfig,
    ResponseReasoningText,
    ResponseUsage,
    ResponsesCreateRequest,
    ResponsesCreateResponse,
)


@dataclass
class ParsedContentBlock:
    """Provider-agnostic intermediate representation of a parsed content block."""

    api_tag: str
    role: str
    content: Any
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
        events = []
        if not self._block_active:
            if block_start := self.block_start(self._current_block_index, api_tag):
                events.extend(block_start)
            self._block_active = True
        return events

    def close_block(self) -> List[dict]:
        events = []
        if self._block_active:
            if block_stop := self.block_stop(self._current_block_index):
                events.extend(block_stop)
            self._block_active = False
            self._current_block_index += 1
        return events

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        return []

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        return []

    def block_stop(self, index: int) -> List[dict]:
        return []

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: str = "stop",
    ) -> List[dict]:
        return []

    def stream_ping(self) -> List[dict]:
        return []

    @property
    def ping_interval_s(self) -> Optional[float]:
        return None

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def current_block_index(self):
        return self._current_block_index

    def create_unique_id(self):
        raise NotImplementedError

    def non_stream_response(
        self,
        parsed_blocks: List["ParsedContentBlock"],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "stop",
    ) -> Any:
        raise NotImplementedError


class OpenAIFormatter(BaseAPIFormatter):
    def create_unique_id(self):
        return "cmpl-" + str(uuid.uuid4().hex)

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        if api_tag == "tool_calls" and isinstance(text, list):
            delta = {api_tag: text, "role": role or "assistant"}
        else:
            delta = {api_tag: str(text) if text is not None else "", "role": role or "assistant"}

        chunk = ChatCompletionResponse(
            id=self.unique_id,
            model=self.model_name,
            object="chat.completion.chunk",
            created=self.created,
            choices=[StreamChoice(index=0, delta=ChoiceDelta(**delta))],
        )

        return [
            {
                "data": json.dumps(
                    chunk.model_dump(exclude_unset=True),
                    default=pydantic_encoder,
                    ensure_ascii=False,
                )
            }
        ]

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: OpenAIStopReason = "stop",
    ) -> List[dict]:
        events = []

        finish_chunk = ChatCompletionResponse(
            id=self.unique_id,
            model=self.model_name,
            object="chat.completion.chunk",
            created=self.created,
            choices=[StreamChoice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason)],
        )
        events.append(
            {
                "data": json.dumps(
                    finish_chunk.model_dump(exclude_unset=True),
                    default=pydantic_encoder,
                    ensure_ascii=False,
                )
            }
        )

        if input_tokens:
            usage_chunk = ChatCompletionResponse(
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
            events.append({"data": json.dumps(usage_chunk.model_dump(exclude_unset=True))})

        events.append({"data": "[DONE]"})
        return events

    def non_stream_response(
        self,
        parsed_blocks: List["ParsedContentBlock"],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "stop",
    ):
        choices = []
        if parsed_blocks:
            message = {"role": parsed_blocks[0].role}
            for block in parsed_blocks:
                if block.api_tag in message:
                    existing = message[block.api_tag]
                    if isinstance(existing, list) and isinstance(block.content, list):
                        message[block.api_tag] = existing + block.content
                    elif isinstance(existing, str) and isinstance(block.content, str):
                        message[block.api_tag] = existing + block.content
                else:
                    message[block.api_tag] = block.content

            choices.append(
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason,
                )
            )

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
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return [{"event": "message_start", "data": json.dumps(data)}]

    def stream_ping(self) -> List[dict]:
        return [{"event": "ping", "data": json.dumps({"type": "ping"})}]

    @property
    def ping_interval_s(self) -> Optional[float]:
        return 5.0

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        self._current_api_tag = api_tag

        if api_tag == "tool_calls":
            return []

        if api_tag == "reasoning":
            content_block = {"type": "thinking", "thinking": ""}
        else:
            block_type = "text" if api_tag == "content" else api_tag
            content_block = {"type": block_type, "text": ""}

        data = {
            "type": "content_block_start",
            "index": self._current_block_index,
            "content_block": content_block,
        }
        return [{"event": "content_block_start", "data": json.dumps(data)}]

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        self._current_api_tag = api_tag

        if api_tag == "tool_calls":
            events = []
            parsed_tools = text if isinstance(text, list) else []
            for tool_call in parsed_tools:
                if hasattr(tool_call, "model_dump"):
                    tool_call = tool_call.model_dump(exclude_none=True)
                function_payload = tool_call.get("function", {})
                arguments = function_payload.get("arguments", "{}")
                tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"

                events.append(
                    {
                        "event": "content_block_start",
                        "data": json.dumps(
                            {
                                "type": "content_block_start",
                                "index": self._current_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": function_payload.get("name", "unknown_tool"),
                                },
                            }
                        ),
                    }
                )
                events.append(
                    {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": self._current_block_index,
                                "delta": {"type": "input_json_delta", "partial_json": str(arguments)},
                            }
                        ),
                    }
                )
                events.append(
                    {
                        "event": "content_block_stop",
                        "data": json.dumps(
                            {"type": "content_block_stop", "index": self._current_block_index}
                        ),
                    }
                )
                self._current_block_index += 1

            return events

        if api_tag == "reasoning":
            delta = {"type": "thinking_delta", "thinking": str(text) if text is not None else ""}
        else:
            delta = {"type": "text_delta", "text": str(text) if text is not None else ""}

        data = {"type": "content_block_delta", "index": self._current_block_index, "delta": delta}
        return [{"event": "content_block_delta", "data": json.dumps(data)}]

    def block_stop(self, index: int) -> List[dict]:
        if self._current_api_tag == "tool_calls":
            return []
        data = {"type": "content_block_stop", "index": self._current_block_index}
        return [{"event": "content_block_stop", "data": json.dumps(data)}]

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: AnthropicStopReason = "end_turn",
    ) -> List[dict]:
        stop_reason = "tool_use" if self._current_api_tag == "tool_calls" else finish_reason
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens or 0},
        }
        return [
            {"event": "message_delta", "data": json.dumps(delta)},
            {"event": "message_stop", "data": json.dumps({"type": "message_stop"})},
        ]

    def non_stream_response(
        self,
        parsed_blocks: List["ParsedContentBlock"],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "end_turn",
    ):
        content_blocks = []
        for block in parsed_blocks:
            if block.api_tag == "reasoning":
                content_blocks.append(AnthropicThinkingBlock(type="thinking", thinking=block.content))
            elif block.api_tag == "content":
                content_blocks.append(AnthropicTextBlock(type="text", text=block.content))
            elif block.api_tag == "tool_calls":
                finish_reason = "tool_use"
                tool_list = block.content if isinstance(block.content, list) else []
                for tool_call in tool_list:
                    if hasattr(tool_call, "model_dump"):
                        tool_call = tool_call.model_dump(exclude_none=True)
                    function_payload = tool_call.get("function", {})
                    arguments = function_payload.get("arguments", "{}")
                    content_blocks.append(
                        AnthropicToolUseBlock(
                            id=tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                            name=function_payload.get("name", "unknown_tool"),
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


class ResponsesFormatter(BaseAPIFormatter):
    def __init__(
        self,
        model_name: str,
        request_model: ResponsesCreateRequest,
        unique_id: Optional[str] = None,
    ):
        super().__init__(model_name, unique_id)
        self.request_model = request_model
        self._current_api_tag: Optional[str] = None
        self._current_items: List[Any] = []
        self._output_items: List[Any] = []
        self._output_index = 0
        self._current_output_start_index = 0

    def create_unique_id(self):
        return "resp_" + str(uuid.uuid4().hex)

    def _json(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, default=pydantic_encoder, ensure_ascii=False)

    def _event(self, name: str, payload: Dict[str, Any]) -> Dict[str, str]:
        return {"event": name, "data": self._json(payload)}

    def _text_payload(self) -> Dict[str, Any]:
        if self.request_model.text:
            return self.request_model.text.model_dump(exclude_none=True, by_alias=True)
        return {"format": {"type": "text"}}

    def _tools_payload(self) -> List[Dict[str, Any]]:
        if not self.request_model.tools:
            return []
        return [tool.model_dump(exclude_none=True) for tool in self.request_model.tools]

    def _tool_choice_payload(self) -> Any:
        if self.request_model.tool_choice is None:
            return "auto"
        if isinstance(self.request_model.tool_choice, str):
            return self.request_model.tool_choice
        return self.request_model.tool_choice.model_dump(exclude_none=True)

    def _usage_payload(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> Optional[ResponseUsage]:
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return ResponseUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    def _response_payload(
        self,
        status: Literal["in_progress", "completed"],
        output: List[Any],
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> ResponsesCreateResponse:
        return ResponsesCreateResponse(
            id=self.unique_id,
            created_at=self.created,
            status=status,
            completed_at=int(time.time()) if status == "completed" else None,
            instructions=self.request_model.instructions,
            max_output_tokens=self.request_model.max_output_tokens,
            model=self.model_name,
            output=output,
            parallel_tool_calls=self.request_model.parallel_tool_calls,
            previous_response_id=self.request_model.previous_response_id,
            reasoning=ResponseReasoningSummaryConfig(
                effort=self.request_model.reasoning.effort if self.request_model.reasoning else None,
                summary=None,
            ),
            store=self.request_model.store,
            temperature=self.request_model.temperature,
            text=self._text_payload(),
            tool_choice=self._tool_choice_payload(),
            tools=self._tools_payload(),
            top_p=self.request_model.top_p,
            truncation=self.request_model.truncation,
            usage=self._usage_payload(input_tokens, output_tokens, total_tokens),
            user=self.request_model.user,
            metadata=self.request_model.metadata,
        )

    def _convert_tool_call(
        self,
        call: Any,
        status: Literal["in_progress", "completed"],
    ) -> ResponseFunctionCallItem:
        if hasattr(call, "model_dump"):
            call = call.model_dump(exclude_none=True)

        function_payload = call.get("function", {}) if isinstance(call, dict) else {}
        call_id = call.get("id") or call.get("call_id") or "call_" + str(uuid.uuid4().hex)

        return ResponseFunctionCallItem(
            call_id=call_id,
            name=function_payload.get("name", ""),
            arguments=function_payload.get("arguments", "{}"),
            status=status,
        )

    def stream_start(self) -> List[dict]:
        response = self._response_payload(status="in_progress", output=[])
        return [
            self._event("response.created", {"type": "response.created", "response": response}),
            self._event("response.in_progress", {"type": "response.in_progress", "response": response}),
        ]

    def block_start(self, index: int, api_tag: str) -> List[dict]:
        self._current_api_tag = api_tag
        self._current_items = []
        self._current_output_start_index = self._output_index

        if api_tag != "content":
            return []

        message_item = ResponseOutputMessage(status="in_progress", content=[ResponseOutputText(text="")])
        self._current_items = [message_item]
        added_item = message_item.model_copy(update={"content": []})

        return [
            self._event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": self._current_output_start_index,
                    "item": added_item,
                },
            ),
            self._event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": message_item.id,
                    "output_index": self._current_output_start_index,
                    "content_index": 0,
                    "part": ResponseOutputText(text=""),
                },
            ),
        ]

    def stream_chunk(self, api_tag: str, text: Any, role: str) -> List[dict]:
        if api_tag == "content":
            if not self._current_items:
                self._current_items = [ResponseOutputMessage(status="in_progress", content=[ResponseOutputText(text="")])]
            message_item = self._current_items[0]
            message_item.content[0].text += str(text) if text is not None else ""
            return [
                self._event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_item.id,
                        "output_index": self._current_output_start_index,
                        "content_index": 0,
                        "delta": str(text) if text is not None else "",
                    },
                )
            ]

        if api_tag == "reasoning":
            if not self._current_items:
                reasoning_item = ResponseReasoningItem(
                    status="in_progress",
                    content=[ResponseReasoningText(text=str(text) if text is not None else "")],
                )
                self._current_items = [reasoning_item]
                return [
                    self._event(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": self._current_output_start_index,
                            "item": reasoning_item.model_copy(update={"content": []}),
                        },
                    )
                ]

            self._current_items[0].content[0].text += str(text) if text is not None else ""
            return []

        if api_tag == "tool_calls" and isinstance(text, list):
            if self._current_items:
                return []

            converted_calls = [self._convert_tool_call(call, status="in_progress") for call in text]
            self._current_items = converted_calls
            return [
                self._event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": self._current_output_start_index + offset,
                        "item": item,
                    },
                )
                for offset, item in enumerate(converted_calls)
            ]

        return []

    def block_stop(self, index: int) -> List[dict]:
        if not self._current_items:
            return []

        events: List[dict] = []

        if self._current_api_tag == "content":
            message_item = self._current_items[0]
            message_item.status = "completed"
            events.extend(
                [
                    self._event(
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "item_id": message_item.id,
                            "output_index": self._current_output_start_index,
                            "content_index": 0,
                            "text": message_item.content[0].text,
                        },
                    ),
                    self._event(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "item_id": message_item.id,
                            "output_index": self._current_output_start_index,
                            "content_index": 0,
                            "part": message_item.content[0],
                        },
                    ),
                    self._event(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": self._current_output_start_index,
                            "item": message_item,
                        },
                    ),
                ]
            )
        else:
            for offset, item in enumerate(self._current_items):
                item.status = "completed"
                events.append(
                    self._event(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": self._current_output_start_index + offset,
                            "item": item,
                        },
                    )
                )

        self._output_items.extend(self._current_items)
        self._output_index += len(self._current_items)
        self._current_items = []
        self._current_api_tag = None
        return events

    def stream_stop(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: str = "stop",
    ) -> List[dict]:
        response = self._response_payload(
            status="completed",
            output=self._output_items,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        return [self._event("response.completed", {"type": "response.completed", "response": response})]

    def final_stream_response(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> ResponsesCreateResponse:
        return self._response_payload(
            status="completed",
            output=self._output_items,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def non_stream_response(
        self,
        parsed_blocks: List["ParsedContentBlock"],
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        finish_reason: str = "stop",
    ):
        output: List[Any] = []
        current_message: Optional[ResponseOutputMessage] = None

        for block in parsed_blocks:
            if block.api_tag == "content":
                if current_message is None:
                    current_message = ResponseOutputMessage(
                        status="completed",
                        content=[ResponseOutputText(text="")],
                    )
                current_message.content[0].text += str(block.content) if block.content is not None else ""
                continue

            if current_message is not None:
                output.append(current_message)
                current_message = None

            if block.api_tag == "reasoning":
                output.append(
                    ResponseReasoningItem(
                        status="completed",
                        content=[ResponseReasoningText(text=str(block.content) if block.content is not None else "")],
                    )
                )
            elif block.api_tag == "tool_calls":
                for call in block.content or []:
                    output.append(self._convert_tool_call(call, status="completed"))
            else:
                output.append(
                    ResponseOutputMessage(
                        status="completed",
                        content=[ResponseOutputText(text=str(block.content) if block.content is not None else "")],
                    )
                )

        if current_message is not None:
            output.append(current_message)

        return self._response_payload(
            status="completed",
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
