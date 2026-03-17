import json

from gallama.api_response.stream_parser_v2 import StreamParserByTag
from gallama.backend.llm.prompt_engine.by_model.default import tool_parser
from gallama.backend.llm.prompt_engine.by_model.gpt_oss import gpt_oss_tool_parser, gpt_oss
from gallama.backend.llm.prompt_engine.by_model.glm4 import glm4_tool_parser
from gallama.backend.llm.prompt_engine.by_model.minimax import minimax_tool_parser
from gallama.backend.llm.prompt_engine.by_model.ministral3 import ministral3_tool_parser
from gallama.backend.llm.prompt_engine.pe_transformers import PromptEngineTransformers
from gallama.backend.llm.prompt_engine.by_model.qwen3 import qwen3_tool_parser
from gallama.backend.llm.prompt_engine.by_model.qwen35 import qwen35_tool_parser
from gallama.backend.llm.prompt_engine.model_special_tag import MODEL_SPECIAL_TAG


def _arguments(parsed_call):
    return json.loads(parsed_call["function"]["arguments"])


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
    assert MODEL_SPECIAL_TAG["ministral3"] is MODEL_SPECIAL_TAG["mistral3"]
    assert MODEL_SPECIAL_TAG["mistral4"] is MODEL_SPECIAL_TAG["mistral3"]


def test_default_tool_parser_supports_multiple_json_objects():
    tool_text = """
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """

    parsed = tool_parser(tool_text)

    assert len(parsed) == 2
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_qwen3_json_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """

    parsed = qwen3_tool_parser(tool_text)

    assert len(parsed) == 2
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_gpt_oss_tool_parser_supports_single_tool_call():
    tool_text = 'get_weather<|channel|>commentary json<|message|>{"city": "Seoul"}'

    parsed = gpt_oss_tool_parser(tool_text)

    assert len(parsed) == 1
    assert parsed[0]["index"] == 0
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
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
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
    assert parsed[0]["function"]["name"] == "get_current_weather"
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
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


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
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
    assert _arguments(parsed[0]) == {"city": "Seoul"}
    assert _arguments(parsed[1]) == {"city": "Tokyo"}


def test_ministral3_tool_parser_supports_multiple_tool_calls():
    tool_text = """
    [TOOL_CALLS]get_weather[ARGS]{"city": "Seoul"}
    [TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}
    """

    parsed = ministral3_tool_parser(tool_text)

    assert len(parsed) == 2
    assert parsed[0]["index"] == 0
    assert parsed[1]["index"] == 1
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


class _ReasoningEffortProbeTokenizer:
    def __init__(self, chat_template, raise_type_error=False):
        self.chat_template = chat_template
        self.raise_type_error = raise_type_error

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if self.raise_type_error:
            raise TypeError("unexpected keyword argument")

        reasoning_effort = kwargs.get("reasoning_effort", "missing")
        return f"{self.chat_template}|reasoning_effort={reasoning_effort}"


def test_transformers_reasoning_effort_probe_detects_supported_template():
    engine = PromptEngineTransformers.__new__(PromptEngineTransformers)
    tokenizer = _ReasoningEffortProbeTokenizer("{{ reasoning_effort }}")

    assert engine._template_supports_reasoning_effort(tokenizer) is True


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
