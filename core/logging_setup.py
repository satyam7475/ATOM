"""
ATOM v14 -- Centralized logging configuration.

Provides a single entry point for setting up structured logging
across all ATOM modules.

Format: timestamp | module | level | message
Outputs to both console (UTF-8) and rotating file.

Security: A PrivacyFilter is applied to all log handlers so
sensitive patterns (API keys, passwords, tokens) are automatically
redacted from log files before they are written to disk.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")

STRUCTURED_FORMAT = (
    "%(asctime)s | %(name)-22s | %(levelname)-5s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_FILE = LOG_DIR / "atom.log"
MAX_BYTES = 2_000_000
BACKUP_COUNT = 3


class _PrivacyLogFilter(logging.Filter):
    """Scrub sensitive patterns from log messages before they hit disk.

    Formats the message first (msg % args), then redacts the combined
    string. This avoids breaking %-format specifiers when the privacy
    filter matches keywords inside format templates (e.g. 'token: %.0fms').
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from context.privacy_filter import redact
            if record.args:
                try:
                    formatted = record.msg % record.args
                    record.msg = redact(formatted)
                    record.args = None
                except (TypeError, ValueError):
                    if isinstance(record.msg, str):
                        record.msg = redact(record.msg)
            elif isinstance(record.msg, str):
                record.msg = redact(record.msg)
        except Exception:
            logging.getLogger("atom.privacy").debug(
                "Privacy filter failed on record", exc_info=True)
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured format for all ATOM modules.

    Should be called once at startup before any other module loads.
    Idempotent: removes existing handlers before adding new ones.
    """
    LOG_DIR.mkdir(exist_ok=True)

    fmt = logging.Formatter(STRUCTURED_FORMAT, datefmt=DATE_FORMAT)
    privacy_filter = _PrivacyLogFilter()

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8",
             errors="replace", closefd=False)
    )
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addFilter(privacy_filter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
