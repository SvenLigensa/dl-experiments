"""A small colored console + file logger.

Kept deliberately minimal. The colored level names make it easy to spot warnings
during a live demo; the file handler gives every run a durable ``train.log`` inside
its Hydra output directory.
"""

from __future__ import annotations

import logging
import os
import sys

_LEVEL_COLORS = {
    "DEBUG": "\033[38;2;76;175;80m",  # green
    "INFO": "\033[38;2;33;150;243m",  # blue
    "WARNING": "\033[38;2;255;152;0m",  # orange
    "ERROR": "\033[38;2;229;57;53m",  # red
    "CRITICAL": "\033[38;2;229;57;53m",  # red
}
_RESET = "\033[0m"

_FMT = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d) %(levelname)s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname)
        if color:
            record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def get_logger(
    name: str = "repro",
    output_dir: str | None = None,
    level: int = logging.INFO,
    is_main: bool = True,
) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name.
        output_dir: If given (and ``is_main``), also write to ``<output_dir>/train.log``.
        level: Logging level.
        is_main: Only the main process prints to the console / writes the log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:  # already configured
        return logger

    if is_main:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_ColoredFormatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(console)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(output_dir, "train.log"), mode="a"
            )
            file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
            logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger
