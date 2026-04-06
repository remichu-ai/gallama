import re
import json
from gallama.logger.logger import logger
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List, Optional


def glm4_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    results = []

    # Clean whitespace to avoid processing empty strings
    if not tool_text or not tool_text.strip():
        return []

    try:
        # 1. Define Regex Patterns

        # PATTERN EXPLANATION:
        # (?P<name>[a-zA-Z0-9_\-]+)  -> Capture the function name (alphanumeric, underscores, dashes)
        # \s* -> Match any whitespace (newlines/spaces) between name and args
        # (?P<args>(?:<arg_key>...)+)-> Capture the entire block of arguments.
        #                               We assume a function call MUST have at least one arg to be detected
        #                               in this format, acting as the anchor.
        tool_pattern = re.compile(
            r'(?P<name>[a-zA-Z0-9_\-]+)\s*(?P<args>(?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)+)',
            re.DOTALL
        )

        # Matches inner: <arg_key>Key</arg_key>...<arg_value>Value</arg_value>
        arg_pattern = re.compile(
            r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>',
            re.DOTALL
        )

        # 2. Find all tool calls
        # This will find "set_title" followed by its args, then "next_function" followed by its args.
        tool_matches = tool_pattern.finditer(tool_text)

        for match in tool_matches:
            tool_name = match.group("name")
            args_block = match.group("args")

            # 3. Extract key-value pairs from the args block
            arguments_dict = {}
            arg_matches = arg_pattern.findall(args_block)

            for key, value_text in arg_matches:
                key = key.strip()
                # value_text might contain newlines (like the joke request), keep it raw or clean it
                # Usually we just strip outer whitespace
                value_text = value_text.strip()

                # Attempt to parse as JSON (for numbers/lists/bools), otherwise keep string
                try:
                    arguments_dict[key] = json.loads(value_text)
                except (json.JSONDecodeError, TypeError):
                    arguments_dict[key] = value_text

            logger.info(f"tool: {tool_name} args: {arguments_dict}")

            parsed_call = ParsedToolCall(
                name=tool_name,
                arguments=arguments_dict,
            )

            results.append(parsed_call)

    except Exception as e:
        logger.error(f"Error parsing tool text: {e}")
        # In a streaming context, you might want to silently return [] if the text is incomplete
        raise e

    return results

def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return "<tool_call>\n"
    else:
        return f"<tool_call>\n{tool_name}\n"

glm4 = {
    "tool": TagDefinition(
        start_marker="<tool_call>",
        end_marker="</tool_call>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=glm4_tool_parser,
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
