from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition
)
from .....utils.utils import get_response_tool_uid
from typing import List, Dict, Optional


def qwen3_tool_parser(tool_text: str, extra_vars: dict = None) -> List[Dict]:
    """
    Parse Qwen JSON-style tool call format.

    Example:
    {"name": "get_weather", "arguments": {"city": "Seoul"}}
    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    """
    results = []
    decoder = json.JSONDecoder()
    pos = 0

    # Initialize state
    if extra_vars is None:
        extra_vars = {"state": {}}
    if not extra_vars.get("state"):
        extra_vars["state"] = {}
    if "tool_call" not in extra_vars["state"]:
        extra_vars["state"]["tool_call"] = 0

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

            chunk_data = ChoiceDeltaToolCall(
                index=extra_vars["state"]["tool_call"],
                id=get_response_tool_uid(),
                function=ChoiceDeltaToolCallFunction(
                    name=tool.get("name"),
                    arguments=json.dumps(tool.get("arguments", ""))
                ),
                type="function"
            )

            results.append(chunk_data.model_dump(exclude_unset=True))
            extra_vars["state"]["tool_call"] += 1

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
