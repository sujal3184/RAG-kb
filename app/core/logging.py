"""Logging setup.

Call `configure_logging()` once, when the app starts. After that, any file
can do:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")

and it will be formatted consistently everywhere.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.config.settings import Settings

# Fields that already exist on every log record — we skip these when
# pulling out "extra" custom fields, so we don't duplicate them.
_STANDARD_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JSONFormatter(logging.Formatter):
    """Turns each log line into a JSON object.

    Why JSON? In production, logs are read by machines (like Grafana), not
    just humans. JSON is easy for those tools to search and filter.
    """

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS:
                data[key] = value

        return json.dumps(data, default=str)


def configure_logging(settings: Settings) -> None:
    """Set up how logs look and where they go (console, for now).

    Args:
        settings: the app's settings, used to pick the log level and format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.value)
    root_logger.handlers.clear()  # avoid duplicate log lines

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter() if settings.LOG_JSON else logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(handler)

    # These libraries are chatty — quiet them down a bit.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)