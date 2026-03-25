import json
import logging

from forecasting_api.logging_config import JsonFormatter, configure_logging, get_logger


def test_configure_logging_returns_singleton():
    left = configure_logging("INFO")
    right = configure_logging("DEBUG")
    assert left is right
    assert right.level == 10


def test_get_logger_returns_child_logger():
    logger = get_logger("service")
    assert logger.name.endswith("service")


def test_json_formatter_includes_optional_fields():
    record = logging.LogRecord(
        name="rulfm.service",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-1"
    record.event_type = "ACCESS"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "req-1"
    assert payload["event_type"] == "ACCESS"
