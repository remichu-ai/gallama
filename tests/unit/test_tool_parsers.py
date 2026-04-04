import json
import asyncio

from gallama.api_response.chat_response import chat_completion_response, chat_completion_response_stream
from gallama.api_response.stream_parser_v2 import StreamParserByTag
from gallama.data_classes.data_class import ChatMLQuery, TagDefinition
from gallama.data_classes.generation_data_class import GenEnd, GenQueueDynamic, GenText, GenerationStats
from gallama.backend.llm.prompt_engine.by_model.default import tool_parser
from gallama.backend.llm.prompt_engine.by_model.gemma4 import gemma4_tool_parser, gemma4
from gallama.backend.llm.prompt_engine.by_model.gpt_oss import gpt_oss_tool_parser, gpt_oss
from gallama.backend.llm.prompt_engine.by_model.glm4 import glm4_tool_parser
from gallama.backend.llm.prompt_engine.by_model.minimax import minimax_tool_parser
from gallama.backend.llm.prompt_engine.by_model.ministral3 import ministral3_tool_parser
from gallama.backend.llm.prompt_engine.pe_transformers import PromptEngineTransformers
from gallama.backend.llm.prompt_engine.by_model.qwen3 import qwen3_tool_parser
from gallama.backend.llm.prompt_engine.by_model.qwen35 import qwen35_tool_parser
from gallama.backend.llm.prompt_engine.model_special_tag import MODEL_SPECIAL_TAG, resolve_vision_token


def _arguments(parsed_call):
    if hasattr(parsed_call, "model_dump"):
        parsed_call = parsed_call.model_dump(exclude_none=True)

    if "function" in parsed_call:
        return json.loads(parsed_call["function"]["arguments"])

    return parsed_call["arguments"]


def _tool_name(parsed_call):
    if hasattr(parsed_call, "model_dump"):
        parsed_call = parsed_call.model_dump(exclude_none=True)

    if "function" in parsed_call:
        return parsed_call["function"]["name"]

    return parsed_call["name"]


def test_model_special_tag_maps_exllamav3_scoped_aliases_to_expected_parsers():
    assert MODEL_SPECIAL_TAG["qwen2"] is MODEL_SPECIAL_TAG["qwen2_5_vl"]
    assert MODEL_SPECIAL_TAG["qwen3"] is MODEL_SPECIAL_TAG["qwen3_moe"]
    assert MODEL_SPECIAL_TAG["qwen3_next"] is MODEL_SPECIAL_TAG["qwen3"]
    assert MODEL_SPECIAL_TAG["qwen3_vl"] is MODEL_SPECIAL_TAG["qwen3"]
    assert MODEL_SPECIAL_TAG["qwen3_vl_moe"] is MODEL_SPECIAL_TAG["qwen3"]
    assert MODEL_SPECIAL_TAG["qwen3_5"] is MODEL_SPECIAL_TAG["qwen3_5_moe"]
    assert MODEL_SPECIAL_TAG["step3p5"] is MODEL_SPECIAL_TAG["qwen3_5"]
    assert MODEL_SPECIAL_TAG["nemotron_h"] is MODEL_SPECIAL_TAG["qwen3_5"]
    assert MODEL_SPECIAL_TAG["gpt_oss"] is gpt_oss
    assert MODEL_SPECIAL_TAG["glm4"] is MODEL_SPECIAL_TAG["glm4_moe"]
    assert MODEL_SPECIAL_TAG["glm4v"] is MODEL_SPECIAL_TAG["glm4v_moe"]
    assert MODEL_SPECIAL_TAG["gemma4"] is gemma4
    assert MODEL_SPECIAL_TAG["ministral3"] is MODEL_SPECIAL_TAG["mistral3"]
    assert MODEL_SPECIAL_TAG["mistral4"] is MODEL_SPECIAL_TAG["mistral3"]


def test_default_tool_parser_supports_multiple_json_objects():
    tool_text = """
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """

    parsed = tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_qwen3_json_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """

    parsed = qwen3_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_gpt_oss_tool_parser_supports_single_tool_call():
    tool_text = 'get_weather<|channel|>commentary json<|message|>{"city": "Seoul"}'

    parsed = gpt_oss_tool_parser(tool_text)

    assert len(parsed) == 1
    assert _tool_name(parsed[0]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}


def test_gpt_oss_stream_parser_supports_multiple_tool_calls_and_mixed_blocks():
    parser = StreamParserByTag(tag_definitions=list(gpt_oss.values()))
    generated = (
        "<|channel|>analysis<|message|>Need weather data<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Seoul\"}<|call|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Tokyo\"}<|call|>"
        "<|start|>assistant<|channel|>final<|message|>Done<|return|>"
    )

    parsed_blocks = parser.parse_full_text(generated)

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["reasoning", "tool_calls", "content"]
    assert parsed_blocks[0][0].post_processor(parsed_blocks[0][1]) == "Need weather data"
    parsed_tools = parsed_blocks[1][0].post_processor(parsed_blocks[1][1])
    assert len(parsed_tools) == 2
    assert _arguments(parsed_tools[0]) == {"city": "Seoul"}
    assert _arguments(parsed_tools[1]) == {"city": "Tokyo"}
    assert parsed_blocks[2][0].post_processor(parsed_blocks[2][1]) == "Done"


def test_qwen35_xml_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    <function=get_weather>
    <parameter=city>Seoul</parameter>
    </function>
    <function=get_weather>
    <parameter=city>Tokyo</parameter>
    </function>
    """

    parsed = qwen35_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_qwen35_xml_tool_parser_supports_nemotron_style_parameters():
    tool_text = """
    <function=get_current_weather>
    <parameter=location>
    Boston, MA
    </parameter>
    <parameter=unit>
    fahrenheit
    </parameter>
    </function>
    """

    parsed = qwen35_tool_parser(tool_text)

    assert len(parsed) == 1
    assert _tool_name(parsed[0]) == "get_current_weather"
    assert _arguments(parsed[0]) == {"location": "Boston, MA", "unit": "fahrenheit"}


def test_glm4_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    get_weather
    <arg_key>city</arg_key><arg_value>"Seoul"</arg_value>
    get_weather
    <arg_key>city</arg_key><arg_value>"Tokyo"</arg_value>
    """

    parsed = glm4_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_gemma4_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    <|tool_call>call:get_weather{city:<|"|>Seoul<|"|>,days:3}<tool_call|>
    <|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>,options:{units:<|"|>metric<|"|>,alerts:true}}<tool_call|>
    """

    parsed = gemma4_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul", "days": 3}
    assert _arguments(parsed[1]) == {"city": "Tokyo", "options": {"units": "metric", "alerts": True}}


def test_gemma4_stream_parser_withholds_trailing_text_after_tool_call():
    parser = StreamParserByTag(tag_definitions=list(gemma4.values()))
    generated = (
        '<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>\n\n'
        'The weather is sunny.'
    )

    parsed_blocks = parser.parse_full_text(generated)

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["tool_calls"]
    parsed_tools = parsed_blocks[0][0].post_processor(parsed_blocks[0][1])
    assert len(parsed_tools) == 1
    assert _arguments(parsed_tools[0]) == {"city": "Seoul"}
    assert parser.generation_should_stop is True


def test_gemma4_stream_parser_allows_consecutive_tool_calls_only():
    parser = StreamParserByTag(tag_definitions=list(gemma4.values()))
    generated = (
        '<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>\n\n'
        '<|tool_call>call:get_weather{"city":"Tokyo"}<tool_call|>'
    )

    parsed_blocks = parser.parse_full_text(generated)

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["tool_calls"]
    parsed_tools = parsed_blocks[0][0].post_processor(parsed_blocks[0][1])
    assert len(parsed_tools) == 2
    assert _arguments(parsed_tools[0]) == {"city": "Seoul"}
    assert _arguments(parsed_tools[1]) == {"city": "Tokyo"}
    assert parser.generation_should_stop is False


def test_stream_parser_allowed_next_tag_accepts_tag_definition_entries():
    next_tag = TagDefinition(
        start_marker="<next>",
        end_marker="</next>",
        tag_type="next_tag",
        api_tag="content",
    )
    first_tag = TagDefinition(
        start_marker="<tool>",
        end_marker="</tool>",
        include_markers=True,
        tag_type="tool_calls",
        api_tag="tool_calls",
        allowed_next_tag=[next_tag],
    )

    parser = StreamParserByTag(tag_definitions=[first_tag, next_tag])
    parsed_blocks = parser.parse_full_text("<tool>call</tool>\n<next>ok</next>")

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["tool_calls", "content"]
    assert parsed_blocks[0][1] == "<tool>call</tool>"
    assert parsed_blocks[1][1] == "ok"
    assert parser.generation_should_stop is False


def test_stream_parser_allowed_next_tag_accepts_literal_prefix_entries():
    first_tag = TagDefinition(
        start_marker="<tool>",
        end_marker="</tool>",
        include_markers=True,
        tag_type="tool_calls",
        api_tag="tool_calls",
        allowed_next_tag=["STOP_HERE"],
    )

    parser = StreamParserByTag(tag_definitions=[first_tag])
    parsed_blocks = parser.parse_full_text("<tool>call</tool>\nSTOP_HERE")

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["tool_calls", "content"]
    assert parsed_blocks[0][1] == "<tool>call</tool>"
    assert parsed_blocks[1][1] == "STOP_HERE"
    assert parser.generation_should_stop is False


def test_stream_parser_allowed_next_tag_treats_strings_as_literal_only():
    next_tag = TagDefinition(
        start_marker="<next>",
        end_marker="</next>",
        tag_type="tool_calls",
        api_tag="content",
    )
    first_tag = TagDefinition(
        start_marker="<tool>",
        end_marker="</tool>",
        include_markers=True,
        tag_type="first_tag",
        api_tag="tool_calls",
        allowed_next_tag=["tool_calls"],
    )

    parser = StreamParserByTag(tag_definitions=[first_tag, next_tag])
    parsed_blocks = parser.parse_full_text("<tool>call</tool>\n<next>ok</next>")

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["tool_calls"]
    assert parsed_blocks[0][1] == "<tool>call</tool>"
    assert parser.generation_should_stop is True


def test_chat_completion_response_omits_gemma4_trailing_text_from_assistant_message():
    async def _run():
        gen_queue = GenQueueDynamic()
        gen_queue.put_nowait(
            GenText(
                content=(
                    '<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>\n\n'
                    'The weather is sunny.'
                )
            )
        )
        gen_queue.put_nowait(
            GenerationStats(
                stop_reason="tool_use",
                input_tokens_count=5,
                output_tokens_count=4,
            )
        )
        gen_queue.put_nowait(GenEnd())

        response = await chat_completion_response(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="openai",
            tag_definitions=list(gemma4.values()),
        )

        message = response.choices[0].message
        assert response.choices[0].finish_reason == "tool_calls"
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "get_weather"
        assert json.loads(message.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert not message.content

    asyncio.run(_run())


def test_minimax_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    <invoke name="get_weather">
    <parameter name="city">"Seoul"</parameter>
    </invoke>
    <invoke name="get_weather">
    <parameter name="city">"Tokyo"</parameter>
    </invoke>
    """

    parsed = minimax_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_ministral3_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    [TOOL_CALLS]get_weather[ARGS]{"city": "Seoul"}
    [TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}
    """

    parsed = ministral3_tool_parser(tool_text)

    assert len(parsed) == 2
    assert _tool_name(parsed[0]) == "get_weather"
    assert _tool_name(parsed[1]) == "get_weather"
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_ministral3_stream_parser_supports_thinking_and_tool_calls():
    parser = StreamParserByTag(tag_definitions=list(MODEL_SPECIAL_TAG["mistral3"].values()))
    generated = (
        "[THINK]Need weather data[/THINK]"
        "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Seoul\"}"
        "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Tokyo\"}</s>"
    )

    parsed_blocks = parser.parse_full_text(generated)

    assert [tag.api_tag for tag, _ in parsed_blocks] == ["reasoning", "tool_calls"]
    assert parsed_blocks[0][1] == "Need weather data"

    parsed_tools = parsed_blocks[1][0].post_processor(parsed_blocks[1][1])
    assert len(parsed_tools) == 2
    assert _arguments(parsed_tools[0]) == {"city": "Seoul"}
    assert _arguments(parsed_tools[1]) == {"city": "Tokyo"}


def test_chat_completion_response_assigns_sequential_tool_call_indexes():
    async def _run():
        gen_queue = GenQueueDynamic()
        gen_queue.put_nowait(
            GenText(
                content=(
                    '<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>\n'
                    '<|tool_call>call:get_weather{"city":"Tokyo"}<tool_call|>'
                )
            )
        )
        gen_queue.put_nowait(
            GenerationStats(
                stop_reason="tool_use",
                input_tokens_count=5,
                output_tokens_count=4,
            )
        )
        gen_queue.put_nowait(GenEnd())

        response = await chat_completion_response(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="openai",
            tag_definitions=list(gemma4.values()),
        )

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert [tool_call.index for tool_call in tool_calls] == [0, 1]
        assert all(tool_call.id for tool_call in tool_calls)

    asyncio.run(_run())


def test_chat_completion_response_stream_assigns_sequential_tool_call_indexes():
    async def _run():
        gen_queue = GenQueueDynamic()
        gen_queue.put_nowait(
            GenText(
                content=(
                    '<|tool_call>call:get_weather{"city":"Seoul"}<tool_call|>\n'
                    '<|tool_call>call:get_weather{"city":"Tokyo"}<tool_call|>'
                )
            )
        )
        gen_queue.put_nowait(
            GenerationStats(
                stop_reason="tool_use",
                input_tokens_count=5,
                output_tokens_count=4,
            )
        )
        gen_queue.put_nowait(GenEnd())

        streamed_tool_calls = []
        async for event in chat_completion_response_stream(
            query=ChatMLQuery.model_validate(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Weather?"}],
                }
            ),
            gen_queue=gen_queue,
            model_name="test-model",
            request=None,
            provider="openai",
            tag_definitions=list(gemma4.values()),
        ):
            data = event.get("data")
            if not data or data == "[DONE]":
                continue

            payload = json.loads(data)
            choices = payload.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            streamed_tool_calls.extend(delta.get("tool_calls", []))

        assert [tool_call["index"] for tool_call in streamed_tool_calls] == [0, 1]
        assert all(tool_call["id"] for tool_call in streamed_tool_calls)

    asyncio.run(_run())


class _ReasoningEffortProbeTokenizer:
    def __init__(self, chat_template, raise_type_error=False):
        self.chat_template = chat_template
        self.raise_type_error = raise_type_error

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if self.raise_type_error:
            raise TypeError("unexpected keyword argument")

        reasoning_effort = kwargs.get("reasoning_effort", "missing")
        return f"{self.chat_template}|reasoning_effort={reasoning_effort}"


class _TemplateProbeTokenizer:
    def __init__(self, chat_template, rendered_prompt):
        self.chat_template = chat_template
        self.rendered_prompt = rendered_prompt

    def apply_chat_template(self, **kwargs):
        return self.rendered_prompt


def test_transformers_reasoning_effort_probe_detects_supported_template():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    tokenizer = _ReasoningEffortProbeTokenizer("{{ reasoning_effort }}")

    assert engine._template_supports_reasoning_effort(tokenizer) is True


class _VisionProbeToken:
    def __init__(self, content):
        self.content = content


class _VisionProbeTokenizer:
    def __init__(self, all_special_tokens=None, special_tokens_map=None, added_tokens_decoder=None):
        self.all_special_tokens = all_special_tokens or []
        self.special_tokens_map = special_tokens_map or {}
        self.added_tokens_decoder = added_tokens_decoder or {}


def test_resolve_vision_token_prefers_explicit_model_mapping():
    tokenizer = _VisionProbeTokenizer(all_special_tokens=["<|image|>"])

    assert resolve_vision_token("glm4v", tokenizer) == "<|begin_of_image|><|image|><|end_of_image|>"


def test_resolve_vision_token_knows_gemma4_placeholder():
    assert resolve_vision_token("gemma4", tokenizer=None) == "<|image|>"


def test_resolve_vision_token_infers_sequence_from_tokenizer_metadata():
    tokenizer = _VisionProbeTokenizer(
        all_special_tokens=["<|vision_bos|>", "<|IMAGE|>", "<|vision_eos|>"],
        added_tokens_decoder={
            1: _VisionProbeToken("<|vision_bos|>"),
            2: _VisionProbeToken("<|IMAGE|>"),
            3: _VisionProbeToken("<|vision_eos|>"),
        },
    )

    assert resolve_vision_token("unknown_model_type", tokenizer) == "<|vision_bos|><|IMAGE|><|vision_eos|>"


def test_transformers_ensure_vision_token_backfills_from_tokenizer_metadata():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    engine.model_type = "unknown_model_type"
    engine._vision_token = None
    engine._transformer_tokenizer = _VisionProbeTokenizer(
        special_tokens_map={
            "additional_special_tokens": [
                "<start_of_image>",
                "<image_soft_token>",
                "<end_of_image>",
            ]
        }
    )

    assert engine.ensure_vision_token() == "<start_of_image><image_soft_token><end_of_image>"


def test_transformers_reasoning_effort_probe_rejects_unsupported_template():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)

    no_template_support = _ReasoningEffortProbeTokenizer("{{ messages }}")
    assert engine._template_supports_reasoning_effort(no_template_support) is False

    no_runtime_support = _ReasoningEffortProbeTokenizer("{{ reasoning_effort }}", raise_type_error=True)
    assert engine._template_supports_reasoning_effort(no_runtime_support) is False


def test_transformers_thinking_detection_handles_mistral_think_tags():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    engine._transformer_tokenizer = _ReasoningEffortProbeTokenizer(
        "{% if block['type'] == 'thinking' %}[THINK]{% endif %}"
    )

    assert engine.check_thinking_model() is True


def test_transformers_get_prompt_does_not_append_extra_closed_gemma4_thought_end_marker():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    engine._transformer_tokenizer = _TemplateProbeTokenizer(
        chat_template="<|think|><|channel>thought",
        rendered_prompt="<bos><|turn>model\n<|channel>thought\n<channel|>",
    )
    engine.thinking_tag = MODEL_SPECIAL_TAG["gemma4"]["thinking"]
    engine.is_thinking_model = True
    engine.support_developer_role = True
    engine.support_list_content = True
    engine.support_reasoning_effort = False
    engine.model_type = "gemma4"
    engine._vision_token = "<|image|>"

    query = ChatMLQuery.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
    )

    prompt, starting_tag = engine.get_prompt(query=query)

    assert prompt == "<bos><|turn>model\n<|channel>thought\n<channel|>"
    assert starting_tag.tag_type == "text"


def test_transformers_appends_structured_output_schema_to_last_user_string_message():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    conversation = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Ack"},
        {"role": "user", "content": "Please rate this"},
    ]
    query = ChatMLQuery.model_validate({
        "messages": [{"role": "user", "content": "ignored"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "rating",
                "schema": {
                    "type": "object",
                    "properties": {
                        "rating": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["rating"],
                },
            },
        },
    })

    engine._append_structured_output_schema_instruction(conversation, query)

    assert conversation[0]["content"] == "First"
    assert conversation[2]["content"].startswith("Please rate this\n\nAnswer using the following schema:\n")
    assert '"rating"' in conversation[2]["content"]
    assert '"minimum": 1' in conversation[2]["content"]


def test_transformers_appends_structured_output_schema_to_last_user_list_message():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Show result"}]}
    ]
    query = ChatMLQuery.model_validate({
        "messages": [{"role": "user", "content": "ignored"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
            },
        },
    })

    engine._append_structured_output_schema_instruction(conversation, query)

    assert conversation[0]["content"][0] == {"type": "text", "text": "Show result"}
    assert conversation[0]["content"][-1]["type"] == "text"
    assert conversation[0]["content"][-1]["text"].startswith("\n\nAnswer using the following schema:\n")
    assert '"ok"' in conversation[0]["content"][-1]["text"]
