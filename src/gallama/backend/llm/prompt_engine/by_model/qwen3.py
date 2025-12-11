from .default import tool_parser
from .....data_classes.data_class import (
    TagDefinition
)

qwen3_moe = {
    "tool": TagDefinition(
        start_marker="<tool_call>",
        end_marker="</tool_call>",
        tag_type="tool_calls",
        api_tag="tool_calls",
        role="tool",
        post_processor=tool_parser,
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