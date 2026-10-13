"""Oriented bounding-box collision proxies used by MiVerse reward predicates."""

from __future__ import annotations

import numpy as np


def local_bbox_world_points(
    vertices,
    position,
    orientation,
    *,
    scale=(1.0, 1.0, 1.0),
    name: str = "local bbox",
) -> np.ndarray:
    """Transform asset-local vertices using a scalar-first (WXYZ) quaternion."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise RuntimeError(
            f"{name} vertices must have shape (N, 3) with N > 0; "
            f"got {points.shape}."
        )
    position = np.asarray(position, dtype=float).reshape(-1)
    orientation = np.asarray(orientation, dtype=float).reshape(-1)
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if position.shape != (3,) or orientation.shape != (4,) or scale.shape != (3,):
        raise RuntimeError(
            f"{name} pose must contain XYZ position, WXYZ orientation and XYZ "
            f"scale; got {position.shape}, {orientation.shape} and {scale.shape}."
        )
    if not (
        np.all(np.isfinite(points))
        and np.all(np.isfinite(position))
        and np.all(np.isfinite(orientation))
        and np.all(np.isfinite(scale))
    ):
        raise RuntimeError(f"{name} vertices or pose contain non-finite values.")

    quat_norm = float(np.linalg.norm(orientation))
    if quat_norm <= np.finfo(float).eps:
        raise RuntimeError(f"{name} orientation quaternion has zero length.")
    w, x, y, z = orientation / quat_norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return (points * scale) @ rotation.T + position


def _convex_hull_xy(points: np.ndarray, *, name: str) -> np.ndarray:
    """Return a counter-clockwise convex hull for world-frame XY points."""
    unique = np.unique(np.asarray(points, dtype=float)[:, :2], axis=0)
    if len(unique) < 3:
        raise RuntimeError(f"{name} XY projection must contain at least 3 points.")

    ordered = sorted(map(tuple, unique.tolist()))

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise RuntimeError(f"{name} XY projection is degenerate.")
    return hull


def oriented_bbox_prisms_intersect(
    vertices_a,
    position_a,
    orientation_a,
    vertices_b,
    position_b,
    orientation_b,
    *,
    scale_a=(1.0, 1.0, 1.0),
    scale_b=(1.0, 1.0, 1.0),
    atol: float = 0.0,
) -> bool:
    """Intersect convex XY projections and world Z intervals, including contact.

    This prism approximation preserves arbitrary yaw and matches MiVerse's
    upright mobile-robot and robot-vacuum collision proxies.
    """
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be a finite non-negative value, got {atol!r}.")

    world_a = local_bbox_world_points(
        vertices_a,
        position_a,
        orientation_a,
        scale=scale_a,
        name="oriented bbox A",
    )
    world_b = local_bbox_world_points(
        vertices_b,
        position_b,
        orientation_b,
        scale=scale_b,
        name="oriented bbox B",
    )
    coordinate_scale = max(
        1.0,
        float(np.abs(world_a).max()),
        float(np.abs(world_b).max()),
    )
    effective_atol = atol + 16.0 * np.finfo(float).eps * coordinate_scale

    if (
        world_a[:, 2].max() < world_b[:, 2].min() - effective_atol
        or world_b[:, 2].max() < world_a[:, 2].min() - effective_atol
    ):
        return False

    hull_a = _convex_hull_xy(world_a, name="oriented bbox A")
    hull_b = _convex_hull_xy(world_b, name="oriented bbox B")
    for hull in (hull_a, hull_b):
        edges = np.roll(hull, -1, axis=0) - hull
        for edge in edges:
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= np.finfo(float).eps:
                continue
            axis /= norm
            projection_a = hull_a @ axis
            projection_b = hull_b @ axis
            if (
                projection_a.max() < projection_b.min() - effective_atol
                or projection_b.max() < projection_a.min() - effective_atol
            ):
                return False
    return True
