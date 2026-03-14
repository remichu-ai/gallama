from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    TagDefinition
)
from .....utils.utils import get_response_tool_uid
from typing import List, Dict


def tool_parser(tool_text: str, extra_vars: dict = None) -> List[Dict]:
    results = []
    decoder = json.JSONDecoder()
    pos = 0

    # Initialize state if needed
    if extra_vars is None:
        extra_vars = {"state": {}}
    if not extra_vars.get("state"):
        extra_vars["state"] = {}
    if "tool_call" not in extra_vars["state"]:
        extra_vars["state"]["tool_call"] = 0

    while pos < len(tool_text):
        # 1. Skip whitespace/newlines to find the start of the next JSON object
        if tool_text[pos].isspace():
            pos += 1
            continue

        try:
            # 2. raw_decode returns the object and the index where it ended
            tool, end_pos = decoder.raw_decode(tool_text, pos)
            pos = end_pos  # Update position for the next iteration

            # 3. Process the found tool
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

            # 4. Add to results and increment the tool index
            results.append(chunk_data.model_dump(exclude_unset=True))
            extra_vars["state"]["tool_call"] += 1

        except json.JSONDecodeError as e:
            # If we hit parsing garbage or incomplete JSON at the end, raise
            raise e

    return results