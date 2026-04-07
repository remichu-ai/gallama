import logging

from gallama.logger.logger import (
    LOG_VERBOSITY_ENV_VAR,
    PlainTextFormatter,
    VerbosityFilter,
    normalize_log_verbosity,
)


def test_normalize_log_verbosity_supports_zero():
    assert normalize_log_verbosity(0) == 0
    assert normalize_log_verbosity(-1) == 0


def test_verbosity_filter_hides_regular_info_logs_at_default_verbosity(monkeypatch):
    monkeypatch.setenv(LOG_VERBOSITY_ENV_VAR, "0")
    verbosity_filter = VerbosityFilter()

    info_record = logging.LogRecord("test", logging.INFO, __file__, 1, "info", (), None)
    basic_info_record = logging.LogRecord("test", logging.INFO, __file__, 1, "basic", (), None)
    basic_info_record.gallama_basic = True
    warning_record = logging.LogRecord("test", logging.WARNING, __file__, 1, "warn", (), None)

    assert verbosity_filter.filter(info_record) is False
    assert verbosity_filter.filter(basic_info_record) is True
    assert verbosity_filter.filter(warning_record) is True


def test_plain_text_formatter_prefixes_request_id_for_non_basic_logs():
    formatter = PlainTextFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
    record.request_id = "abcd1234"

    assert formatter.format(record) == "[req:abcd1234] hello"


def test_plain_text_formatter_skips_request_prefix_for_basic_logs():
    formatter = PlainTextFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "REQ abcd1234 GET /health", (), None)
    record.request_id = "abcd1234"
    record.gallama_basic = True

    assert formatter.format(record) == "REQ abcd1234 GET /health"
