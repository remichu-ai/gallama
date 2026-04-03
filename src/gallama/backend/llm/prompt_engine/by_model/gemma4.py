import json
import re
from typing import Dict, List, Optional

from gallama.logger.logger import logger

from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition,
)
from .....utils.utils import get_response_tool_uid


GEMMA4_TOOL_START_MARKER = "<|tool_call>"
GEMMA4_TOOL_END_MARKER = "<tool_call|>"
GEMMA4_THINKING_START_MARKER = "<|channel>thought\n"
GEMMA4_THINKING_END_MARKER = "<channel|>"
GEMMA4_ESCAPE_TOKEN = '<|"|>'

GEMMA4_TOOL_PATTERN = re.compile(
    rf"{re.escape(GEMMA4_TOOL_START_MARKER)}\s*call:(?P<name>[A-Za-z0-9_.-]+)\s*(?P<body>.*?)\s*{re.escape(GEMMA4_TOOL_END_MARKER)}",
    re.DOTALL,
)


def _normalize_gemma4_arguments(arguments_text: str) -> str:
    # Gemma 4 sometimes emits arguments in a relaxed object syntax:
    # {shape:<|"|>square<|"|>,count:2}
    # Convert that into normal JSON before handing it to json.loads.
    normalized = arguments_text

    # Turn Gemma's custom escaped strings into standard JSON strings.
    normalized = re.sub(
        rf"{re.escape(GEMMA4_ESCAPE_TOKEN)}(.*?){re.escape(GEMMA4_ESCAPE_TOKEN)}",
        lambda match: json.dumps(match.group(1)),
        normalized,
        flags=re.DOTALL,
    )

    # Quote bare object keys so `shape:` becomes `"shape":`.
    normalized = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)',
        r'\1"\2"\3',
        normalized,
    )

    return normalized


def _parse_gemma4_arguments(arguments_text: str) -> Dict:
    try:
        # Fast path: valid JSON output from the model.
        return json.loads(arguments_text)
    except json.JSONDecodeError:
        # Fallback: normalize Gemma's relaxed syntax, then parse as JSON.
        normalized = _normalize_gemma4_arguments(arguments_text)
        return json.loads(normalized)


def gemma4_tool_parser(tool_text: str, extra_vars: dict = None) -> List[Dict]:
    """
    Example tool call formats:
    <|tool_call>call:calculate_geometry{"shape": "square", "elements": "circles"}<tool_call|>
    <|tool_call>call:calculate_geometry{shape:<|"|>square<|"|>,elements:<|"|>circles<|"|>}<tool_call|>
    """

    results = []

    if extra_vars is None:
        extra_vars = {"state": {}}
    if not extra_vars.get("state"):
        extra_vars["state"] = {}
    if "tool_call" not in extra_vars["state"]:
        extra_vars["state"]["tool_call"] = 0

    if not tool_text or not tool_text.strip():
        return []

    try:
        # Parse one or more tool calls from a single generated chunk.
        matches = GEMMA4_TOOL_PATTERN.finditer(tool_text)

        for match in matches:
            tool_name = match.group("name").strip()
            tool_args_str = match.group("body").strip()

            arguments_dict = _parse_gemma4_arguments(tool_args_str)
            logger.info(f"tool: {tool_name} args: {arguments_dict}")

            # Match the same streaming tool-call payload shape used by other parsers.
            chunk_data = ChoiceDeltaToolCall(
                index=extra_vars["state"]["tool_call"],
                id=get_response_tool_uid(),
                function=ChoiceDeltaToolCallFunction(
                    name=tool_name,
                    arguments=json.dumps(arguments_dict),
                ),
                type="function",
            )

            results.append(chunk_data.model_dump(exclude_unset=True))
            extra_vars["state"]["tool_call"] += 1

    except Exception as e:
        logger.error(f"Error parsing Gemma 4 tool calls: {e}")
        raise e

    return results


def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return f"{GEMMA4_TOOL_START_MARKER}call:"
    else:
        return f"{GEMMA4_TOOL_START_MARKER}call:{tool_name}" + "{"


gemma4 = {
    "tool": TagDefinition(
        start_marker=GEMMA4_TOOL_START_MARKER,
        end_marker=GEMMA4_TOOL_END_MARKER,
        include_markers=True,
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=gemma4_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True,
    ),
    "thinking": TagDefinition(
        start_marker=GEMMA4_THINKING_START_MARKER,
        end_marker=GEMMA4_THINKING_END_MARKER,
        role="assistant",
        allowed_roles={"assistant", "tool"},
        tag_type="thinking",
        api_tag="reasoning",
    ),
}
