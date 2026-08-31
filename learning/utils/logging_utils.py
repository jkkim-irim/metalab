"""Logging + number-formatting utilities (de-lerobot'd).

Faithful copies of ``lerobot.utils.utils.init_logging`` and ``format_big_number`` for the calls
ALLEX makes (``init_logging(accelerator=accelerator)``, ``format_big_number(n)``). The log line
format and the K/M/B/... suffix table match LeRobot 0.4.4 exactly.
"""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any


def init_logging(
    log_file: Path | None = None,
    display_pid: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    accelerator: Any | None = None,
) -> None:
    """Initialize root-logger config.

    In multi-GPU training only the main process logs to console (avoids duplicate output); non-main
    processes are silenced on console but may still log to file. Mirrors LeRobot's behaviour and
    log-line format.
    """

    def custom_format(record: logging.LogRecord) -> str:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        pid_str = f"[PID: {os.getpid()}] " if display_pid else ""
        return f"{record.levelname} {pid_str}{dt} {fnameline[-15:]:>15} {record.getMessage()}"

    formatter = logging.Formatter()
    formatter.format = custom_format

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Clear any existing handlers.
    logger.handlers.clear()

    # Determine if this is a non-main process in distributed training.
    is_main_process = accelerator.is_main_process if accelerator is not None else True

    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level.upper())
        logger.addHandler(console_handler)
    else:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level.upper())
        logger.addHandler(file_handler)


def format_big_number(num: float, precision: int = 0) -> str:
    """Format a big number with a K/M/B/T/Q suffix (LeRobot ``format_big_number``)."""
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num
