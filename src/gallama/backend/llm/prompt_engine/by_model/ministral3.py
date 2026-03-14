from gallama.logger.logger import logger
import json
import re
from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition
)
from .....utils.utils import get_response_tool_uid
from typing import List, Dict, Optional


def ministral3_tool_parser(tool_text: str, extra_vars: dict = None) -> List[Dict]:
    """
    Example of tool call format:
    [TOOL_CALLS]get_weather[ARGS]{"city": "Seoul"}[TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}

    Parsing Logic:
    1. Identify blocks starting with [TOOL_CALLS]
    2. Extract the name up to [ARGS]
    3. Extract the JSON payload up to the next [TOOL_CALLS] or End of String.
    """

    results = []

    # Initialize state (same as your reference)
    if extra_vars is None:
        extra_vars = {"state": {}}
    if not extra_vars.get("state"):
        extra_vars["state"] = {}
    if "tool_call" not in extra_vars["state"]:
        extra_vars["state"]["tool_call"] = 0

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

            # Construct the response object
            chunk_data = ChoiceDeltaToolCall(
                index=extra_vars["state"]["tool_call"],
                id=get_response_tool_uid(),
                function=ChoiceDeltaToolCallFunction(
                    name=tool_name,
                    # Dump it back to string to strictly conform to ChoiceDeltaToolCallFunction requirements
                    arguments=json.dumps(arguments_dict)
                ),
                type="function"
            )

            results.append(chunk_data.model_dump(exclude_unset=True))
            extra_vars["state"]["tool_call"] += 1

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
}