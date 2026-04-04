from gallama.logger.logger import logger
import json
import re
from .....data_classes.data_class import (
    ParsedToolCall,
    TagDefinition
)
from typing import List, Optional


GPT_OSS_TOOL_PATTERN = re.compile(
    r'(?P<name>[A-Za-z0-9_.-]+)<\|channel\|>commentary(?:\s+[A-Za-z0-9_.-]+)?<\|message\|>(?P<body>.*?)(?=[A-Za-z0-9_.-]+<\|channel\|>commentary(?:\s+[A-Za-z0-9_.-]+)?<\|message\|>|$)',
    re.DOTALL
)


def gpt_oss_tool_parser(tool_text: str, extra_vars: dict = None) -> List[ParsedToolCall]:
    results = []

    if not tool_text or not tool_text.strip():
        return []

    matches = list(GPT_OSS_TOOL_PATTERN.finditer(tool_text.strip()))
    if not matches:
        return []

    for match in matches:
        tool_name = match.group("name").strip()
        tool_args_str = match.group("body").strip()

        try:
            arguments_dict = json.loads(tool_args_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse GPT-OSS JSON for tool '{tool_name}'. content: {tool_args_str}")
            continue

        logger.info(f"tool: {tool_name} args: {arguments_dict}")

        parsed_call = ParsedToolCall(
            name=tool_name,
            arguments=arguments_dict,
        )

        results.append(parsed_call)
    return results


def gpt_oss_text_post_processor(text: str, extra_args: dict = None) -> str:
    if not text:
        return ""
    return text.replace("<|return|>", "").replace("<|end|>", "")


def tool_prompt(tool_name: Optional[str] = None):
    if tool_name is None:
        return " to=functions."
    else:
        return f" to=functions.{tool_name}<|channel|>commentary json<|message|>"


gpt_oss = {
    "tool": TagDefinition(
        start_marker=r'(?:<\|start\|>assistant)?\s+to=functions\.',
        end_marker="<|call|>",
        marker_type="regex",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="assistant",
        post_processor=gpt_oss_tool_parser,
        prompt_init=tool_prompt,
        wait_till_complete=True
    ),
    "thinking": TagDefinition(
        start_marker=r'(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>',
        end_marker="<|end|>",
        marker_type="regex",
        role="assistant",
        allowed_roles={"assistant", "tool"},
        tag_type="thinking",
        api_tag="reasoning",
        post_processor=gpt_oss_text_post_processor,
    ),
    "final": TagDefinition(
        start_marker=r'(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>',
        end_marker="<|return|>",
        marker_type="regex",
        tag_type="text",
        api_tag="content",
        role="assistant",
        post_processor=gpt_oss_text_post_processor,
    ),
}
