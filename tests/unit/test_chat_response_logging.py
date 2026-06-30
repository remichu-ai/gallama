import logging

from gallama.api_response.chat_response import format_generation_stats_log, format_stream_start_log
from gallama.data_classes import GenerationStats
from gallama.data_classes.data_class import TagDefinition
from gallama.logger import logger


def test_format_generation_stats_log_uses_compact_field_names():
    stats = GenerationStats(
        input_tokens_count=10,
        output_tokens_count=20,
        time_to_first_token=1.0,
        time_generate=2.0,
        measured_time_generate=0.5,
        cached_tokens=3,
        cached_pages=1,
        accepted_draft_tokens=4,
        rejected_draft_tokens=2,
    )

    message = format_generation_stats_log("model", stats)

    assert (
        "model | gen 10.0 tok/s | wall_gen 40.0 tok/s | pref 10.0 tok/s | "
        "in 10 | out 20 | tot 30 | ttft 1.00s | gen_t 2.00s | total_t 3.00s | "
        "stop end_turn | cache_tok 3 | cache_pg 1 | draft_acc 4 | draft_rej 2 | "
        "draft_hit 4/20 (20.0%)"
    ) == message


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
