"""MiVerse control-group routing for per-step tool IK."""

from collections.abc import Mapping
from copy import deepcopy

import numpy as np


class ToolIKPlanner:
    """Expose one physical robot's Pink solver through named control groups."""

    backend = "pink"
    ik_backend = "pink"

    def __init__(
        self,
        *,
        name,
        implementation,
        dt,
        joints,
        tool_frames,
        owner,
        tool_robots,
        control_groups,
    ):
        self.name = str(name)
        self.implementation = implementation
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("Planner dt must be positive and finite.")
        self.joints = [str(name) for name in joints]
        self.tool_frames = [str(name) for name in tool_frames]
        if not self.joints or not self.tool_frames:
            raise ValueError("Planner requires non-empty joints and tool_frames.")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError(f"Planner joints must be unique: {self.joints}.")
        if len(set(self.tool_frames)) != len(self.tool_frames):
            raise ValueError(f"Planner tool frames must be unique: {self.tool_frames}.")
        self.owner = owner
        self.tool_robots = {str(frame): robot for frame, robot in tool_robots.items()}
        self.control_groups = {}
        for raw_name, raw_group in control_groups.items():
            name = str(raw_name)
            group_joints = [str(value) for value in raw_group.get("joints") or []]
            group_frames = [str(value) for value in raw_group.get("tool_frames") or []]
            if not group_joints or not group_frames:
                raise ValueError(f"Control group {name!r} requires joints and tool_frames.")
            missing_joints = [value for value in group_joints if value not in self.joints]
            missing_frames = [value for value in group_frames if value not in self.tool_frames]
            if missing_joints or missing_frames:
                raise ValueError(
                    f"Control group {name!r} is outside planner {self.name!r}: "
                    f"joints={missing_joints}, frames={missing_frames}."
                )
            self.control_groups[name] = {"name": name, "joints": group_joints, "tool_frames": group_frames}
        if not self.control_groups:
            raise ValueError("Planner requires at least one control group.")
        self._closed = False

    def get_control_group(self, name):
        try:
            return self.control_groups[str(name)]
        except KeyError:
            raise KeyError(
                f"Unknown control group {name!r} for planner {self.name!r}; "
                f"available groups: {list(self.control_groups)}."
            ) from None

    def resolve_control_group(self, target_frames, *, group_name=None, default_group=None):
        """Select MiVerse's exact frame match, or its smallest unique superset."""
        frames = [str(frame) for frame in target_frames]
        target_set = set(frames)
        if group_name is not None:
            group = self.get_control_group(group_name)
        elif default_group is not None and len(frames) == 1:
            group = self.get_control_group(default_group)
        else:
            candidates = [group for group in self.control_groups.values() if set(group["tool_frames"]) == target_set]
            if not candidates:
                supersets = [
                    group for group in self.control_groups.values() if target_set.issubset(set(group["tool_frames"]))
                ]
                if supersets:
                    extra = min(len(set(group["tool_frames"]) - target_set) for group in supersets)
                    candidates = [group for group in supersets if len(set(group["tool_frames"]) - target_set) == extra]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot infer one control group for tool frames {frames}; "
                    f"candidates={[group['name'] for group in candidates]}. Pass control_group explicitly."
                )
            group = candidates[0]
        unknown = [frame for frame in frames if frame not in group["tool_frames"]]
        if unknown:
            raise ValueError(
                f"Control group {group['name']!r} does not contain target frames "
                f"{unknown}; configured frames are {group['tool_frames']}."
            )
        return group

    def normalize_targets(self, target_poses):
        """Validate the frame-to-pose mapping supplied by NavEval."""
        if not isinstance(target_poses, Mapping) or not target_poses:
            raise ValueError("Tool targets must be a non-empty frame-to-pose mapping.")
        targets = {}
        for name, value in target_poses.items():
            frame = str(name)
            if frame not in self.tool_frames:
                raise ValueError(f"Unknown tool target {frame!r}; available frames are {self.tool_frames}.")
            pose = np.asarray(value, dtype=np.float64)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise ValueError(f"Target pose for {frame!r} must be finite shape (7,), got {pose.shape}.")
            targets[frame] = pose
        return targets

    def solve(self, *, current_qpos, target_poses, root_pose, control_group, env_idx=0):
        """Solve all active joints and select the group's executable columns."""
        if self._closed:
            raise RuntimeError(f"Planner {self.name!r} is closed.")
        targets = self.normalize_targets(target_poses)
        group = self.resolve_control_group(targets, group_name=control_group)
        raw = self.implementation.solve_differential_ik(
            self._full_joint_vector(current_qpos), targets, real_robot_pose=root_pose, env_idx=env_idx
        )
        result = {} if raw is None else deepcopy(raw)
        result.setdefault("status", "Fail")
        result.update(
            backend=self.backend,
            planner_backend=self.backend,
            ik_backend=self.ik_backend,
            planner_name=self.name,
            control_group=group["name"],
            tracked_tool_frames=list(result.get("tracked_tool_frames") or targets),
            command_dt=self.dt,
        )
        if result["status"] != "Success":
            result["joint_names"] = list(group["joints"])
            return result

        names = list(result.get("joint_names") or self.joints)
        missing = [name for name in group["joints"] if name not in names]
        if missing:
            raise ValueError(f"Planner result is missing control-group joints {missing}; result joints are {names}.")
        indices = [names.index(name) for name in group["joints"]]
        for row_key, command_key in (
            ("position", "joint_value"),
            ("velocity", "joint_velocity"),
            ("acceleration", "joint_acceleration"),
        ):
            value = result.get(command_key)
            if value is None:
                if command_key == "joint_value":
                    raise ValueError("Successful Pink result is missing joint_value.")
                value = np.zeros(len(names), dtype=np.float64)
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (len(names),):
                raise ValueError(f"Planner {command_key} shape {array.shape} does not match joints {names}.")
            result[command_key] = array[indices].copy()
            result[row_key] = result[command_key].reshape(1, -1).copy()
        result["joint_names"] = list(group["joints"])
        return result

    def _full_joint_vector(self, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.shape != (len(self.joints),) or not np.all(np.isfinite(array)):
            raise ValueError(f"current_qpos must be finite shape ({len(self.joints)},), got {array.shape}.")
        return array.copy()

    def reset(self, current_qpos=None, *, env_idx=None):
        self.implementation.reset(
            None if current_qpos is None else self._full_joint_vector(current_qpos), env_idx=env_idx
        )

    def close(self):
        if not self._closed:
            self.implementation.close()
            self._closed = True
