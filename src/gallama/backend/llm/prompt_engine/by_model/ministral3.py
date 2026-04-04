from gallama.logger.logger import logger
import json
import re
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

    # Regex Pattern Explanation:
    # \[TOOL_CALLS\]        -> Matches the literal start tag
    # \s*(?P<name>.*?)\s* -> Captures the function name (group 'name'), handles whitespace
    # \[ARGS\]              -> Matches the literal separator tag
    # \s*(?P<args>.*?)      -> Captures the JSON string (group 'args') non-greedily
    # (?=\[TOOL_CALLS\]|$)  -> Positive Lookahead: stops matching when it sees the
    #                          NEXT [TOOL_CALLS] tag or the End of the String ($).
    pattern = r"\[TOOL_CALLS\]\s*(?P<name>.*?)\s*\[ARGS\]\s*(?P<args>.*?)(?=\[TOOL_CALLS\]|$)"

    try:
        # re.DOTALL allows the dot (.) to match newlines, supporting multi-line JSON
        matches = re.finditer(pattern, tool_text, re.DOTALL)

        for match in matches:
            tool_name = match.group("name").strip()
            tool_args_str = match.group("args").strip()

            # Validate and format arguments
            try:
                # We parse it to ensure it is valid JSON
                arguments_dict = json.loads(tool_args_str)
            except json.JSONDecodeError:
                # If the model outputs malformed JSON, we log it.
                # Depending on strictness, you might want to skip this item or raise an error.
                logger.warning(f"Failed to parse JSON for tool '{tool_name}'. content: {tool_args_str}")
                continue

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
        include_markers=True,
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
