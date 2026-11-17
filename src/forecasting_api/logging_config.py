from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        event_type = getattr(record, "event_type", None)
        if event_type:
            payload["event_type"] = event_type
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None) -> logging.Logger:
    logger = logging.getLogger("rulfm")
    raw_level = level if level is not None else os.getenv("RULFM_LOG_LEVEL", "INFO")
    desired_level = raw_level.upper()
    if logger.handlers:
        logger.setLevel(desired_level)
        for handler in logger.handlers:
            handler.setFormatter(JsonFormatter())
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(desired_level)
    logger.propagate = False
    return logger


def get_logger(name: str = "rulfm") -> logging.Logger:
    root = configure_logging()
    return root if name == "rulfm" else root.getChild(name)
