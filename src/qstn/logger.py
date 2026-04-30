"""Logging helpers for QSTN.

QSTN is a library, so importing it should not configure global logging or write
to the console. Applications can opt in with `configure_logging`.
"""

from __future__ import annotations

import logging
import sys

from tqdm.auto import tqdm

PACKAGE_LOGGER_NAME = "qstn"


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm without breaking progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm_write(self.format(record))
        except Exception:
            self.handleError(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger or one of its children.

    Args:
        name: Optional child logger name. Full `qstn.*` names are accepted.

    Returns:
        The requested logger instance.
    """
    if name is None:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name == PACKAGE_LOGGER_NAME or name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    *,
    use_tqdm: bool = True,
    force: bool = False,
) -> None:
    """Configure simple console logging for QSTN.

    Args:
        level: Minimum level emitted by the package logger.
        use_tqdm: If True, console records are written through `tqdm.write`.
        force: If True, replace existing non-null QSTN handlers.
    """
    logger = get_logger()
    configured_handlers = [
        handler for handler in logger.handlers if not isinstance(handler, logging.NullHandler)
    ]

    if configured_handlers and not force:
        logger.setLevel(level)
        for handler in configured_handlers:
            handler.setLevel(level)
        return

    if force:
        logger.handlers = [
            handler for handler in logger.handlers if isinstance(handler, logging.NullHandler)
        ]

    handler: logging.Handler
    if use_tqdm:
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def tqdm_write(message: str) -> None:
    """Write explicit user-facing output without colliding with tqdm bars."""
    tqdm.write(message)


get_logger().addHandler(logging.NullHandler())
