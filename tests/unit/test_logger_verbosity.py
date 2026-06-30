import logging
import sys

from gallama.logger.logger import (
    FILE_LOG_VERBOSITY_ENV_VAR,
    LOG_VERBOSITY_ENV_VAR,
    PlainTextFormatter,
    VerbosityFilter,
    get_logger,
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


def test_plain_text_formatter_includes_exception_traceback_with_request_id():
    formatter = PlainTextFormatter()

    try:
        raise TypeError("tuple indices must be integers or slices, not str")
    except TypeError:
        record = logging.LogRecord("test", logging.ERROR, __file__, 1, "boom", (), None)
        record.request_id = "abcd1234"
        record.exc_info = sys.exc_info()

    formatted = formatter.format(record)

    assert formatted.startswith("[req:abcd1234] boom\nTraceback")
    assert "TypeError: tuple indices must be integers or slices, not str" in formatted


def test_file_verbosity_env_allows_debug_file_logs_with_quiet_default(monkeypatch, tmp_path):
    monkeypatch.setenv(LOG_VERBOSITY_ENV_VAR, "0")
    monkeypatch.setenv(FILE_LOG_VERBOSITY_ENV_VAR, "2")

    log_path = tmp_path / "gallama.log"
    test_logger = get_logger(
        name="test_file_verbosity_env_allows_debug_file_logs_with_quiet_default",
        log_file=str(log_path),
        to_console=False,
        to_file=True,
        to_zmq=False,
    )

    test_logger.debug("debug detail")
    for handler in test_logger.handlers:
        handler.flush()
        handler.close()

    assert "debug detail" in log_path.read_text()
