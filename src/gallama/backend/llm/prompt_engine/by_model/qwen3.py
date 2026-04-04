from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List, Optional


def qwen3_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    """
    Parse Qwen JSON-style tool call format.

    Example:
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """
    results = []
    decoder = json.JSONDecoder()
    pos = 0

    if not tool_text or not tool_text.strip():
        return []

    try:
        while pos < len(tool_text):
            if tool_text[pos].isspace():
                pos += 1
                continue

            tool, end_pos = decoder.raw_decode(tool_text, pos)
            pos = end_pos

            logger.info(f"tool: {tool}")

            parsed_call = ParsedToolCall(
                name=tool.get("name"),
                arguments=tool.get("arguments", ""),
            )

            results.append(parsed_call)

    except Exception as e:
        raise e

    return results


def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return '\n<tool_call>\n{"name": "'
    else:
        return f'\n<tool_call>\n{{"name": "{tool_name}", "arguments": '


qwen3_moe = {
    "tool": TagDefinition(
        start_marker="<tool_call>",
        end_marker="</tool_call>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=qwen3_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True
    ),
    "thinking": TagDefinition(
        start_marker="<think>",
        end_marker="</think>",
        tag_type="thinking",
        role="assistant",
        allowed_roles={"assistant", "tool"},
        api_tag="reasoning"
    )
}
