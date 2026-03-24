import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .data_class import (
    BaseMessage,
    ChatMLQuery,
    FunctionCall,
    JsonSchemaSpec,
    MultiModalAudioContent,
    MultiModalImageContent,
    MultiModalTextContent,
    ParameterSpec,
    ResponseFormat,
    ResponseFormatJSONSchema,
    SingleFunctionDict,
    ToolCall,
    ToolForce,
    ToolSpec,
    FunctionSpec,
)


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class ResponseTextFormatText(BaseModel):
    type: Literal["text"] = "text"


class ResponseTextFormatJSONSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Literal["json_schema"] = "json_schema"
    name: str
    schema_: Dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = None
    description: Optional[str] = None


class ResponseTextConfig(BaseModel):
    format: Union[ResponseTextFormatText, ResponseTextFormatJSONSchema] = Field(
        default_factory=ResponseTextFormatText
    )


class ResponseReasoningConfig(BaseModel):
    effort: Optional[Literal["minimal", "low", "medium", "high"]] = None


class ResponseFunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = True


class ResponseToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    name: str


class ResponseInputContentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: Optional[str] = None
    image_url: Optional[Union[str, Dict[str, Any]]] = None
    file_url: Optional[str] = None
    detail: Optional[Literal["low", "high"]] = None
    audio: Optional[str] = None
    input_audio: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    id: Optional[str] = None
    status: Optional[str] = None
    output: Optional[Any] = None
    content: Optional[List["ResponseInputContentItem"]] = None


class ResponseInputItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "message"
    role: Optional[Literal["system", "developer", "user", "assistant", "tool"]] = None
    content: Optional[Union[str, List[ResponseInputContentItem]]] = None
    call_id: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[Any] = None
    status: Optional[str] = None
    summary: Optional[List[ResponseInputContentItem]] = None


class ResponsesCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: Union[str, ResponseInputItem, List[Union[str, ResponseInputItem]]]
    instructions: Optional[str] = None
    tools: Optional[List[ResponseFunctionTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], ResponseToolChoiceFunction]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    store: bool = False
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user: Optional[str] = None
    truncation: Optional[str] = "disabled"
    reasoning: Optional[ResponseReasoningConfig] = None
    text: Optional[ResponseTextConfig] = None

    def _serialize_json(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _convert_text_format(self) -> Optional[Union[ResponseFormat, ResponseFormatJSONSchema]]:
        if self.text is None:
            return None

        format_spec = self.text.format
        if isinstance(format_spec, ResponseTextFormatText):
            return ResponseFormat(type="text")

        return ResponseFormatJSONSchema(
            type="json_schema",
            json_schema=JsonSchemaSpec(
                name=format_spec.name,
                schema=format_spec.schema_,
                strict=format_spec.strict,
            ),
        )

    def _convert_tools(self) -> Optional[List[ToolSpec]]:
        if not self.tools:
            return None

        converted_tools: List[ToolSpec] = []
        for tool in self.tools:
            if tool.type != "function":
                raise ValueError(
                    f"Unsupported Responses API tool type '{tool.type}'. "
                    "The basic rollout only supports function tools."
                )

            parameters = tool.parameters or {"type": "object", "properties": {}, "required": []}
            converted_tools.append(
                ToolSpec(
                    type="function",
                    function=FunctionSpec(
                        name=tool.name,
                        description=tool.description,
                        parameters=ParameterSpec(
                            type=parameters.get("type", "object"),
                            properties=parameters.get("properties"),
                            required=parameters.get("required", []),
                        ),
                        strict=True if tool.strict is None else tool.strict,
                    ),
                )
            )

        return converted_tools

    def _convert_tool_choice(self) -> Optional[Union[str, ToolForce]]:
        if self.tool_choice is None:
            return None

        if isinstance(self.tool_choice, str):
            return self.tool_choice

        return ToolForce(
            type="function",
            function=SingleFunctionDict(name=self.tool_choice.name),
        )

    def _convert_message_content(
        self,
        content: Optional[Union[str, List[ResponseInputContentItem]]],
    ) -> tuple[Union[str, List[Any]], Optional[str], Optional[List[ToolCall]]]:
        if content is None:
            return "", None, None

        if isinstance(content, str):
            return content, None, None

        multimodal_content: List[Any] = []
        reasoning_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for part in content:
            if part.type in {"input_text", "output_text", "text"}:
                multimodal_content.append(MultiModalTextContent(type="text", text=part.text or ""))
            elif part.type in {"input_image", "image_url"}:
                image_payload = part.image_url
                if isinstance(image_payload, dict):
                    image_payload = image_payload.get("url")
                image_payload = image_payload or part.file_url
                if image_payload:
                    multimodal_content.append(
                        MultiModalImageContent(
                            type="image_url",
                            image_url=MultiModalImageContent.ImageDetail(
                                url=image_payload,
                                detail=part.detail or "high",
                            ),
                        )
                    )
            elif part.type == "input_audio":
                audio_data = part.audio
                if audio_data is None and part.input_audio:
                    audio_data = part.input_audio.get("data") or part.input_audio.get("audio")
                if audio_data:
                    multimodal_content.append(MultiModalAudioContent(type="audio", audio=audio_data))
            elif part.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=part.call_id or part.id or _make_id("call"),
                        function=FunctionCall(
                            name=part.name or "",
                            arguments=part.arguments or "{}",
                        ),
                        type="function",
                    )
                )
            elif part.type in {"reasoning_text", "summary_text"} and part.text:
                reasoning_parts.append(part.text)
            elif part.text is not None:
                multimodal_content.append(MultiModalTextContent(type="text", text=part.text))

        if multimodal_content and all(isinstance(part, MultiModalTextContent) for part in multimodal_content):
            message_content: Union[str, List[Any]] = "".join(part.text for part in multimodal_content)
        else:
            message_content = multimodal_content or ""

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        return message_content, reasoning, tool_calls or None

    def _convert_input_item(self, item: Union[str, ResponseInputItem]) -> List[BaseMessage]:
        if isinstance(item, str):
            return [BaseMessage(role="user", content=item)]

        if item.type == "function_call_output":
            return [
                BaseMessage(
                    role="tool",
                    tool_call_id=item.call_id,
                    content=self._serialize_json(item.output),
                )
            ]

        if item.type == "function_call":
            return [
                BaseMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=item.call_id or item.id or _make_id("call"),
                            function=FunctionCall(
                                name=item.name or "",
                                arguments=item.arguments or "{}",
                            ),
                            type="function",
                        )
                    ],
                )
            ]

        if item.type == "reasoning":
            reasoning_parts: List[str] = []
            for part in item.content or []:
                if part.text:
                    reasoning_parts.append(part.text)
            for part in item.summary or []:
                if part.text:
                    reasoning_parts.append(part.text)
            return [BaseMessage(role="assistant", content="", reasoning="\n".join(reasoning_parts))]

        if item.type != "message" and item.role is None:
            raise ValueError(f"Unsupported Responses API input item type '{item.type}'")

        message_content, reasoning, tool_calls = self._convert_message_content(item.content)
        return [
            BaseMessage(
                role=item.role or "user",
                content=message_content,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )
        ]

    def to_input_messages(self, include_instructions: bool = True) -> List[BaseMessage]:
        messages: List[BaseMessage] = []

        if include_instructions and self.instructions:
            messages.append(BaseMessage(role="system", content=self.instructions))

        input_items = self.input if isinstance(self.input, list) else [self.input]
        for item in input_items:
            messages.extend(self._convert_input_item(item))

        return messages

    def to_chat_ml_query(self, messages: Optional[List[BaseMessage]] = None) -> ChatMLQuery:
        if messages is None:
            messages = self.to_input_messages()

        query_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "tools": self._convert_tools(),
            "tool_choice": self._convert_tool_choice(),
            "max_tokens": self.max_output_tokens,
            "store": self.store,
            "response_format": self._convert_text_format(),
            "reasoning_effort": self.reasoning.effort if self.reasoning else None,
            "parallel_tool_calls": self.parallel_tool_calls,
        }

        filtered_kwargs = {key: value for key, value in query_kwargs.items() if value is not None}
        return ChatMLQuery(**filtered_kwargs)


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[Any] = Field(default_factory=list)


class ResponseReasoningText(BaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ResponseSummaryText(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseOutputMessage(BaseModel):
    id: str = Field(default_factory=lambda: _make_id("msg"))
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: List[ResponseOutputText] = Field(default_factory=list)


class ResponseFunctionCallItem(BaseModel):
    id: str = Field(default_factory=lambda: _make_id("fc"))
    type: Literal["function_call"] = "function_call"
    call_id: str = Field(default_factory=lambda: _make_id("call"))
    name: str
    arguments: str
    status: Literal["in_progress", "completed"] = "completed"


class ResponseReasoningItem(BaseModel):
    id: str = Field(default_factory=lambda: _make_id("rs"))
    type: Literal["reasoning"] = "reasoning"
    summary: Optional[List[ResponseSummaryText]] = None
    content: Optional[List[ResponseReasoningText]] = None
    status: Literal["in_progress", "completed"] = "completed"


class ResponseUsageInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class ResponseUsageOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    input_tokens_details: ResponseUsageInputTokensDetails = Field(
        default_factory=ResponseUsageInputTokensDetails
    )
    output_tokens: int = 0
    output_tokens_details: ResponseUsageOutputTokensDetails = Field(
        default_factory=ResponseUsageOutputTokensDetails
    )
    total_tokens: int = 0


class ResponseReasoningSummaryConfig(BaseModel):
    effort: Optional[Literal["minimal", "low", "medium", "high"]] = None
    summary: Optional[List[ResponseSummaryText]] = None


class ResponsesCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: _make_id("resp"))
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    status: Literal["in_progress", "completed"] = "completed"
    completed_at: Optional[int] = None
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[Union[ResponseOutputMessage, ResponseFunctionCallItem, ResponseReasoningItem]] = Field(
        default_factory=list
    )
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: ResponseReasoningSummaryConfig = Field(default_factory=ResponseReasoningSummaryConfig)
    store: bool = False
    temperature: Optional[float] = None
    text: Dict[str, Any] = Field(default_factory=lambda: {"format": {"type": "text"}})
    tool_choice: Optional[Any] = "auto"
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    top_p: Optional[float] = None
    truncation: Optional[str] = "disabled"
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


ResponseInputContentItem.model_rebuild()


def response_output_to_assistant_messages(
    output_items: List[Union[ResponseOutputMessage, ResponseFunctionCallItem, ResponseReasoningItem, Dict[str, Any]]],
) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    reasoning_parts: List[str] = []
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    def flush_current_message():
        nonlocal reasoning_parts, text_parts, tool_calls
        has_reasoning = any(part for part in reasoning_parts)
        has_text = any(part for part in text_parts)
        has_tools = bool(tool_calls)
        if not (has_reasoning or has_text or has_tools):
            return

        messages.append(
            BaseMessage(
                role="assistant",
                content="".join(text_parts),
                reasoning="\n".join(part for part in reasoning_parts if part) or None,
                tool_calls=tool_calls or None,
            )
        )
        reasoning_parts = []
        text_parts = []
        tool_calls = []

    for output_item in output_items or []:
        item = (
            output_item.model_dump(exclude_none=True, by_alias=True)
            if hasattr(output_item, "model_dump")
            else dict(output_item)
        )
        item_type = item.get("type")

        if item_type == "reasoning":
            for part in item.get("content") or []:
                if hasattr(part, "model_dump"):
                    part = part.model_dump(exclude_none=True, by_alias=True)
                text = part.get("text")
                if text:
                    reasoning_parts.append(text)
            for part in item.get("summary") or []:
                if hasattr(part, "model_dump"):
                    part = part.model_dump(exclude_none=True, by_alias=True)
                text = part.get("text")
                if text:
                    reasoning_parts.append(text)
            continue

        if item_type == "message":
            for part in item.get("content") or []:
                if hasattr(part, "model_dump"):
                    part = part.model_dump(exclude_none=True, by_alias=True)
                if part.get("type") == "output_text" and part.get("text") is not None:
                    text_parts.append(part["text"])
            continue

        if item_type == "function_call":
            tool_calls.append(
                ToolCall(
                    id=item.get("call_id") or item.get("id") or _make_id("call"),
                    function=FunctionCall(
                        name=item.get("name", ""),
                        arguments=item.get("arguments", "{}"),
                    ),
                    type="function",
                )
            )
            continue

        flush_current_message()

    flush_current_message()
    return messages
