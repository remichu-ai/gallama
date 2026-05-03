from fastapi import HTTPException

from .data_classes import ChatMLQuery, ToolForce


def validate_api_request(query: ChatMLQuery):
    if query.temperature is not None and query.temperature <= 0:
        query.temperature = 0.01

    if query.tools:
        if query.tool_choice is None:
            if query.tools:
                query.tool_choice = "auto"
            else:
                query.tool_choice = "none"
        elif query.tool_choice == "none":
            query.tools = None
        elif isinstance(query.tool_choice, ToolForce):
            force_single_tool = []
            for tool in query.tools:
                if tool.function.name == query.tool_choice.function.name:
                    force_single_tool.append(tool)

            if len(force_single_tool) != 1:
                raise Exception("tool_choice not exist in function list")
            else:
                query.tools = force_single_tool
    else:
        query.tool_choice = "none"

    if query.prefix_strings and (query.regex_pattern or query.regex_prefix_pattern):
        raise HTTPException(
            status_code=400,
            detail="refix_strings and regex_pattern, regex_prefix_pattern can not be used together",
        )

    return query
