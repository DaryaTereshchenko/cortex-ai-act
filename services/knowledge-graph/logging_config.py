"""
Centralized logging configuration for the knowledge-graph service.

Usage:
    from logging_config import configure_logging
    configure_logging()  # Call once at application startup
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = Path(__file__).parent / "logs"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 3


def configure_logging(
    *,
    level: int = logging.INFO,
    log_dir: Path | None = None,
    log_file: str = "app.log",
    console: bool = True,
    file: bool = True,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """
    Configure logging with console and/or file output.

    Args:
        level: Console log level (default: INFO).
        log_dir: Directory for log files (default: ./logs).
        log_file: Name of the log file (default: app.log).
        console: Enable console output (default: True).
        file: Enable file output (default: True).
        log_format: Log message format.
        date_format: Timestamp format.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated files to keep.
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (rotating)
    if file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
