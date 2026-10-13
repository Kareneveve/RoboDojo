"""Validate the optional root-frame envelope used by collision rewards."""

from collections.abc import Mapping

import numpy as np


def parse_configured_collision_bbox(config, *, config_path: str):
    """Return eight root-frame vertices, or ``None`` when not configured."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_path} must be a mapping.")
    if config.get("frame") != "root":
        raise ValueError(f"{config_path}.frame must be 'root', got {config.get('frame')!r}.")

    try:
        bounds = np.asarray([config["lower"], config["upper"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_path} must define numeric lower/upper XYZ bounds.") from exc
    if bounds.shape != (2, 3):
        raise ValueError(f"{config_path} lower/upper must each contain three values; got shape {bounds.shape}.")
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"{config_path} bounds must be finite.")
    lower, upper = bounds
    if np.any(lower >= upper):
        raise ValueError(f"{config_path} lower bounds must be strictly below upper bounds; got {bounds.tolist()}.")

    vertices = np.asarray(
        [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
        dtype=float,
    )
    vertices.setflags(write=False)
    return vertices
