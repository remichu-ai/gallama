import logging

from gallama.api_response.chat_response import format_stream_start_log
from gallama.data_classes.data_class import TagDefinition
from gallama.logger import logger


def test_format_stream_start_log_uses_tag_type_at_info_level():
    original_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        tag = TagDefinition(tag_type="text", api_tag="content")
        assert format_stream_start_log(tag) == "Stream starts with text"
        assert format_stream_start_log("tool") == "Stream starts with tool"
    finally:
        logger.setLevel(original_level)


def test_format_stream_start_log_uses_full_repr_at_debug_level():
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        tag = TagDefinition(tag_type="text", api_tag="content")
        message = format_stream_start_log(tag)
        assert message.startswith("Stream starts with ")
        assert "tag_type='text'" in message
        assert message != "Stream starts with text"
    finally:
        logger.setLevel(original_level)
