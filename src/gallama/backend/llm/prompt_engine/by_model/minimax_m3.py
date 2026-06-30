import json
import re
from html import unescape
from typing import Any, List, Optional

from gallama.logger.logger import logger
from .....data_classes.data_class import ParsedToolCall, TagDefinition


MINIMAX_M3_NS_TOKEN = "]<]minimax[>["
MINIMAX_M3_TOOL_START_MARKER = f"{MINIMAX_M3_NS_TOKEN}<tool_call>"
MINIMAX_M3_TOOL_END_MARKER = f"{MINIMAX_M3_NS_TOKEN}</tool_call>"
MINIMAX_M3_THINKING_START_MARKER = "<mm:think>"
MINIMAX_M3_THINKING_END_MARKER = "</mm:think>"

_ELEMENT_PATTERN = re.compile(
    r"<(?P<name>[A-Za-z_][\w.-]*)>(?P<value>.*?)</(?P=name)>",
    re.DOTALL,
)
_INVOKE_PATTERN = re.compile(
    r'<invoke\s+name="(?P<name>[^"]+)">\s*(?P<body>.*?)\s*</invoke>',
    re.DOTALL,
)


def _strip_minimax_m3_namespace(text: str) -> str:
    return (
        text.replace(MINIMAX_M3_TOOL_START_MARKER, "")
        .replace(MINIMAX_M3_TOOL_END_MARKER, "")
        .replace(MINIMAX_M3_NS_TOKEN, "")
    )


def _parse_scalar(raw_value: str) -> Any:
    value_text = unescape(raw_value.strip())
    try:
        return json.loads(value_text)
    except (json.JSONDecodeError, TypeError):
        return value_text


def _merge_xml_value(target: dict, key: str, value: Any) -> None:
    if key not in target:
        target[key] = value
        return

    if not isinstance(target[key], list):
        target[key] = [target[key]]
    target[key].append(value)


def _parse_xml_value(raw_value: str) -> Any:
    matches = list(_ELEMENT_PATTERN.finditer(raw_value))
    if not matches:
        return _parse_scalar(raw_value)

    parsed: dict[str, Any] = {}
    for match in matches:
        _merge_xml_value(parsed, match.group("name"), _parse_xml_value(match.group("value")))

    if set(parsed.keys()) == {"item"}:
        return parsed["item"] if isinstance(parsed["item"], list) else [parsed["item"]]

    return parsed


def minimax_m3_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    """
    Parse MiniMax-M3 namespaced XML tool calls rendered by the HF chat template:

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="tool-name">
    ]<]minimax[>[<param>value]<]minimax[>[</param>
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>
    """
    del extra_vars

    if not tool_text or not tool_text.strip():
        return []

    cleaned_text = _strip_minimax_m3_namespace(tool_text)
    results = []

    for invoke_match in _INVOKE_PATTERN.finditer(cleaned_text):
        tool_name = invoke_match.group("name").strip()
        invoke_body = invoke_match.group("body")

        arguments_dict = {}
        for param_match in _ELEMENT_PATTERN.finditer(invoke_body):
            _merge_xml_value(
                arguments_dict,
                param_match.group("name").strip(),
                _parse_xml_value(param_match.group("value")),
            )

        logger.info(f"tool: {tool_name} args: {arguments_dict}")
        results.append(ParsedToolCall(name=tool_name, arguments=arguments_dict))

    return results


def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return f"{MINIMAX_M3_TOOL_START_MARKER}\n"
    return f'{MINIMAX_M3_TOOL_START_MARKER}\n{MINIMAX_M3_NS_TOKEN}<invoke name="{tool_name}">'


minimax_m3 = {
    "tool": TagDefinition(
        start_marker=MINIMAX_M3_TOOL_START_MARKER,
        end_marker=MINIMAX_M3_TOOL_END_MARKER,
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=minimax_m3_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True,
    ),
    "thinking": TagDefinition(
        start_marker=MINIMAX_M3_THINKING_START_MARKER,
        end_marker=MINIMAX_M3_THINKING_END_MARKER,
        role="assistant",
        allowed_roles={"assistant", "tool"},
        tag_type="thinking",
        api_tag="reasoning",
    ),
}
