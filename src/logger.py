"""Centralised logging setup for the whole project.

Every module should obtain its logger via :func:`get_logger` instead of using
bare ``print`` statements.  This keeps all output formatted consistently and
lets you see *exactly* which stage of the pipeline produced a message.

Example:
    >>> from src.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Chunk %s processed", chunk_id)

The first call to :func:`get_logger` configures the root logger (idempotently),
so you never need to call :func:`setup_logging` yourself.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# ``sys.stdout`` keeps logs on the same stream the pipeline already uses,
# making ordering predictable when the app is run from a terminal.
_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
_CONSOLE_HANDLER.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_DEFAULT_LOG_FILE = _LOGS_DIR / "pipeline.log"


def setup_logging(
    level: int = logging.INFO,
    *,
    log_file: str | Path | None = _DEFAULT_LOG_FILE,
    force: bool = False,
) -> None:
    """Configure the root logger once.

    Args:
        level: Minimum severity to emit (default ``INFO``).
        log_file: Optional path to write logs to.  ``None`` disables the file
            handler entirely; the default is ``logs/pipeline.log`` relative to
            the project root.
        force: Re-run configuration even if the logger was already configured.
            Mainly useful for tests that want a different log level.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return

    root.setLevel(level)
    for handler in root.handlers:
        root.removeHandler(handler)
    root.addHandler(_CONSOLE_HANDLER)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    logging.getLogger().info("Logging initialised (level=%s, file=%s)",
                             logging.getLevelName(level),
                             Path(log_file).resolve() if log_file else "none")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` that inherits the root configuration.
    """
    setup_logging()
    return logging.getLogger(name)
