from gallama.logger.logger import logger
import json
import re
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List, Optional


def qwen35_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    """
    Parse Qwen 3.5 XML-style tool call format:

    <function=tool_name>
    <parameter=param_key_1>
    param_value_1
    </parameter>
    </function>
    """

    results = []

    if not tool_text or not tool_text.strip():
        return []

    try:
        function_pattern = re.compile(
            r'<function=([^>]+)>\s*(.*?)\s*</function>',
            re.DOTALL
        )

        for func_match in function_pattern.finditer(tool_text):
            tool_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            param_pattern = re.compile(
                r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>',
                re.DOTALL
            )

            arguments_dict = {}
            for param_match in param_pattern.finditer(func_body):
                key = param_match.group(1).strip()
                value_text = param_match.group(2).strip()

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
        raise e

    return results


def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return '\n<tool_call>\n<function='
    else:
        return f'\n<tool_call>\n<function={tool_name}>\n'


qwen35 = {
    "tool": TagDefinition(
        start_marker="<tool_call>",
        end_marker="</tool_call>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=qwen35_tool_parser,
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
