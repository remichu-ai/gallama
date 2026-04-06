from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List, Optional


def ministral3_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    """
    Example of tool call format:
    [TOOL_CALLS]get_weather[ARGS]{"city": "Seoul"}[TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}

    Parsing Logic:
    1. Identify blocks starting with [TOOL_CALLS]
    2. Extract the name up to [ARGS]
    3. Extract the JSON payload up to the next [TOOL_CALLS] or End of String.
    """

    results = []

    # Clean whitespace to avoid parsing errors on empty strings
    if not tool_text or not tool_text.strip():
        return []

    decoder = json.JSONDecoder()
    pos = 0

    try:
        while pos < len(tool_text):
            while pos < len(tool_text) and tool_text[pos].isspace():
                pos += 1

            if pos >= len(tool_text):
                break

            if tool_text.startswith("[TOOL_CALLS]", pos):
                pos += len("[TOOL_CALLS]")
                continue

            args_idx = tool_text.find("[ARGS]", pos)
            if args_idx == -1:
                break

            tool_name = tool_text[pos:args_idx].strip()
            pos = args_idx + len("[ARGS]")

            while pos < len(tool_text) and tool_text[pos].isspace():
                pos += 1

            # Validate and format arguments
            try:
                arguments_dict, pos = decoder.raw_decode(tool_text, pos)
            except json.JSONDecodeError:
                # If the model outputs malformed JSON, we log it.
                # Depending on strictness, you might want to skip this item or raise an error.
                logger.warning(f"Failed to parse JSON for tool '{tool_name}'. content: {tool_text[pos:].strip()}")
                break

            logger.info(f"tool: {tool_name} args: {arguments_dict}")

            parsed_call = ParsedToolCall(
                name=tool_name,
                arguments=arguments_dict,
            )

            results.append(parsed_call)

    except Exception as e:
        logger.error(f"Error parsing Ministral3 tool calls: {e}")
        raise e

    return results

def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return "[TOOL_CALLS]"
    else:
        return f"[TOOL_CALLS]{tool_name}[ARGS]"

ministral3 = {
    "tool": TagDefinition(
        start_marker="[TOOL_CALLS]",
        end_marker="</s>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=ministral3_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True
    ),
    "thinking": TagDefinition(
        start_marker="[THINK]",
        end_marker="[/THINK]",
        tag_type="thinking",
        role="assistant",
        allowed_roles={"assistant", "tool"},
        api_tag="reasoning"
    ),
}
