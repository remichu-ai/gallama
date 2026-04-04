from gallama.logger.logger import logger
import json
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List


def tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    results = []
    decoder = json.JSONDecoder()
    pos = 0

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

            parsed_call = ParsedToolCall(
                name=tool.get("name"),
                arguments=tool.get("arguments", ""),
            )

            results.append(parsed_call)

        except json.JSONDecodeError as e:
            # If we hit parsing garbage or incomplete JSON at the end, raise
            raise e

    return results
