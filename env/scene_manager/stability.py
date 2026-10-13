"""Pose limits used while settling saved evaluation layouts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityLimits:
    angle_deg: float = 30.0
    horizontal_m: float = 0.04
    downward_m: float = 0.04
    upward_m: float = 0.04
    support_drop_m: float = 0.05


def pose_is_stable(first_pos, first_rot, current_pos, current_rot, *, support_height, limits):
    """Compare XY distance, asymmetric Z settling and wxyz rotation limits."""
    first_position = np.asarray(first_pos, dtype=float).reshape(3)
    current_position = np.asarray(current_pos, dtype=float).reshape(3)
    first_quaternion = np.asarray(first_rot, dtype=float).reshape(4)
    current_quaternion = np.asarray(current_rot, dtype=float).reshape(4)
    first_norm = np.linalg.norm(first_quaternion)
    current_norm = np.linalg.norm(current_quaternion)
    if first_norm == 0 or current_norm == 0:
        rotation_deg = float("inf")
    else:
        cosine = abs(float(np.dot(first_quaternion / first_norm, current_quaternion / current_norm)))
        rotation_deg = float(np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))
    delta = current_position - first_position
    return not (
        rotation_deg > limits.angle_deg
        or np.linalg.norm(delta[:2]) > limits.horizontal_m
        or delta[2] < -limits.downward_m
        or delta[2] > limits.upward_m
        or current_position[2] < float(support_height) - limits.support_drop_m
    )


def layout_pose_reference(layout_object, *, current_position, current_orientation):
    """Keep warm-up motion from becoming the saved layout's stability baseline."""
    position = layout_object.get("default_pos")
    orientation = layout_object.get("default_ori")
    return (
        np.asarray(current_position if position is None else position, dtype=float).reshape(3).copy(),
        np.asarray(current_orientation if orientation is None else orientation, dtype=float).reshape(4).copy(),
    )
