from gallama.logger.logger import logger
import json
import re
from html import unescape
from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition
)
from .....utils.utils import get_response_tool_uid
from typing import List, Dict, Optional


def minimax_tool_parser(tool_text: str, extra_vars: dict = None) -> List[Dict]:
    """
    Example of tool call format:
    < minimax: tool_call >
    < invoke
    name = "tool-name-1" >
    < parameter
    name = "param-key-1" > param - value - 1 < / parameter >
    < parameter
    name = "param-key-2" > param - value - 2 < / parameter >
    ...
    < / invoke >
    < / minimax: tool_call >
    """

    results = []

    # Initialize state
    if extra_vars is None:
        extra_vars = {"state": {}}
    if not extra_vars.get("state"):
        extra_vars["state"] = {}
    if "tool_call" not in extra_vars["state"]:
        extra_vars["state"]["tool_call"] = 0

    # Clean whitespace to avoid parsing errors on empty strings
    if not tool_text or not tool_text.strip():
        return []

    invoke_pattern = re.compile(
        r'<invoke\s+name="(?P<name>[^"]+)">\s*(?P<body>.*?)\s*</invoke>',
        re.DOTALL,
    )
    parameter_pattern = re.compile(
        r'<parameter\s+name="(?P<name>[^"]+)">\s*(?P<value>.*?)\s*</parameter>',
        re.DOTALL,
    )

    def parse_argument(raw_value: str):
        value_text = unescape(raw_value.strip())
        try:
            return json.loads(value_text)
        except (json.JSONDecodeError, TypeError):
            return value_text

    try:
        for invoke_match in invoke_pattern.finditer(tool_text):
            tool_name = invoke_match.group("name").strip()
            invoke_body = invoke_match.group("body")

            arguments_dict = {}
            for param_match in parameter_pattern.finditer(invoke_body):
                key = param_match.group("name").strip()
                arguments_dict[key] = parse_argument(param_match.group("value"))

            logger.info(f"tool: {tool_name} args: {arguments_dict}")

            chunk_data = ChoiceDeltaToolCall(
                index=extra_vars["state"]["tool_call"],
                id=get_response_tool_uid(),
                function=ChoiceDeltaToolCallFunction(
                    name=tool_name,
                    arguments=json.dumps(arguments_dict)
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
        return "<minimax:tool_call>"
    else:
        return f'<minimax:tool_call>\n<invoke name="{tool_name}">'

minimax = {
    "tool": TagDefinition(
        start_marker="<minimax:tool_call>",
        end_marker="</minimax:tool_call>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=minimax_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True
    ),
    "thinking": TagDefinition(
        start_marker="<think>",
        end_marker="</think>",
        role="assistant",
        allowed_roles={"assistant", "tool"},
        tag_type="thinking",
        api_tag="reasoning"
    )
}
