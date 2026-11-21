"""Validation helpers for evaluation identifiers and output paths."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=,-]*$")


def safe_path_component(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    """Return a path component that cannot escape its designated directory."""
    text = "" if value is None else str(value)
    if allow_empty and text == "":
        return text
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"{field_name} must be a simple path component, got {text!r}")
    return text


def make_run_id() -> str:
    """Create a collision-resistant identifier for one evaluation invocation."""
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_{os.getpid()}"


def validate_resume_save_dir(candidate: object, expected_parent: Path) -> Path:
    """Accept only a run directory directly below the expected eval folder."""
    candidate_path = Path(str(candidate)).expanduser().resolve(strict=False)
    parent = expected_parent.expanduser().resolve(strict=False)
    if candidate_path.parent != parent:
        raise ValueError(
            f"resume save_dir must be directly below {parent}, got {candidate_path}"
        )
    safe_path_component(candidate_path.name, "resume save_dir")
    return candidate_path
