from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition
)
from .....utils.utils import get_response_tool_uid
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET


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

    try:
        # 1. Wrap the potentially multi-root XML in a fake root tag
        #    This allows us to handle "<invoke>...</invoke><invoke>...</invoke>"
        wrapped_xml = f"<root>{tool_text}</root>"
        root = ET.fromstring(wrapped_xml)

        # 2. Iterate through all <invoke> tags found
        for invoke_item in root.findall('invoke'):
            tool_name = invoke_item.get("name")

            # 3. Aggregate parameters into a single dictionary
            arguments_dict = {}
            for param in invoke_item.findall('parameter'):
                key = param.get("name")
                value_text = param.text if param.text else ""

                # Attempt to parse the inner text as JSON (e.g., lists, numbers, booleans)
                # If it's just a raw string not formatted as JSON, keep it as string.
                try:
                    arguments_dict[key] = json.loads(value_text)
                except (json.JSONDecodeError, TypeError):
                    arguments_dict[key] = value_text

            logger.info(f"tool: {tool_name} args: {arguments_dict}")

            # 4. Construct the response object
            chunk_data = ChoiceDeltaToolCall(
                index=extra_vars["state"]["tool_call"],
                id=get_response_tool_uid(),
                function=ChoiceDeltaToolCallFunction(
                    name=tool_name,
                    # Convert the aggregated dict back to a JSON string for the response format
                    arguments=json.dumps(arguments_dict)
                ),
                type="function"
            )

            results.append(chunk_data.model_dump(exclude_unset=True))
            extra_vars["state"]["tool_call"] += 1

    except ET.ParseError as e:
        # This might happen if the XML is incomplete (streaming mid-tag)
        # Depending on your logic, you might want to suppress this or raise it.
        raise e
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