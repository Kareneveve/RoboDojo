"""Structured console + file logging helpers for RoboDojo VLM Rubric evaluation."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys

LOGGER_NAME = "robodojo.vlm_rubric"


def format_duration(seconds: float) -> str:
    """Format an elapsed duration in seconds into a compact human-readable string."""
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, rem = divmod(total, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{rem:02.0f}s"


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that ignores BrokenPipeError when stdout is piped (e.g. `| head`)."""

    def handleError(self, record: logging.LogRecord) -> None:
        _, exc_value, _ = sys.exc_info()
        if isinstance(exc_value, BrokenPipeError):
            return
        super().handleError(record)


def _update_latest_symlink(log_path: Path) -> None:
    latest_link = log_path.parent / "latest.log"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        os.symlink(log_path.name, latest_link)
    except OSError:
        # Symlinks may be unsupported on some filesystems; ignore cleanly.
        pass


def configure_rubric_logging(
    *,
    log_file: str | Path | None = None,
    log_dir: str | Path | None = None,
    run_name: str = "vlm_rubric",
    level: str = "INFO",
    quiet: bool = False,
) -> tuple[logging.Logger, Path | None]:
    """Configure console and optional file handlers for the VLM Rubric logger."""

    logger = logging.getLogger(LOGGER_NAME)
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(min(numeric_level, logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    console = _SafeStreamHandler(sys.stdout)
    console.setLevel(logging.WARNING if quiet else numeric_level)
    console.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    resolved_log_path: Path | None = None
    if log_file is not None:
        resolved_log_path = Path(log_file).expanduser().resolve()
    elif log_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name).strip("_") or "rubric"
        resolved_log_path = Path(log_dir).expanduser().resolve() / f"{timestamp}_{safe_name}.log"

    if resolved_log_path is not None:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
        file_handler.setLevel(min(numeric_level, logging.DEBUG if numeric_level == logging.DEBUG else logging.INFO))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)
        _update_latest_symlink(resolved_log_path)

    return logger, resolved_log_path


__all__ = ["LOGGER_NAME", "configure_rubric_logging", "format_duration"]
