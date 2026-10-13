"""MiVerse per-step multi-tool IK for evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pink import Configuration, solve_ik
from pink.limits.limit import Limit
from pink.tasks import DampingTask, FrameTask, PostureTask
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import qpsolvers
import transforms3d as t3d
import yaml


class _JointPositionWindowLimit(Limit):
    """Constrain the integrated position to a per-request joint window."""

    def __init__(self, lower_q, upper_q):
        self.lower_q = np.asarray(lower_q, dtype=np.float64).reshape(-1)
        self.upper_q = np.asarray(upper_q, dtype=np.float64).reshape(-1)
        if self.lower_q.shape != self.upper_q.shape:
            raise ValueError("lower_q and upper_q must share the same shape.")

    def compute_qp_inequalities(self, configuration, dt):
        q_curr = np.asarray(
            configuration.q if hasattr(configuration, "q") else configuration,
            dtype=np.float64,
        ).reshape(-1)
        if q_curr.shape != self.lower_q.shape:
            return None
        eye = np.eye(q_curr.size, dtype=np.float64)
        return (
            np.vstack((eye, -eye)),
            np.hstack((self.upper_q - q_curr, q_curr - self.lower_q)),
        )


def _kinematics_from_yaml(yml_path) -> dict:
    if yml_path is None:
        return {}
    path = Path(yml_path).expanduser()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return dict(data.get("robot_cfg", {}).get("kinematics", {}) or {})


def _locked_joint_positions(kinematics: Mapping) -> dict[str, float]:
    return {str(name): float(value) for name, value in dict(kinematics.get("lock_joints", {}) or {}).items()}


class PinkPlanner:
    """Integrate one measured-state IK step for the configured tool frames.

    Keeps MiVerse's posture, command-continuity and joint-limit behavior.
    M92UW evaluation disables self-collision checking in its robot config.
    """

    def __init__(
        self,
        robot_origin_pose,
        active_joints_name,
        urdf_path,
        mesh_dir=None,
        dt=None,
        yml_path=None,
        *,
        tool_frames=None,
        locked_joint_positions: Mapping[str, float] | None = None,
        null_space_weight_overrides: Mapping[str, float] | None = None,
        solver: str | None = None,
        position_cost: float = 2.0,
        orientation_cost: float = 2.0,
        frame_task_gain: float = 0.8,
        frame_task_lm_damping: float = 1.0e-4,
        posture_task_cost: float = 1.0e-2,
        null_space_task_cost: float = 1.0e-2,
        previous_command_task_cost: float = 0.0,
        previous_command_reset_threshold: float = 0.25,
        damping_task_cost: float = 1.0e-4,
        qp_damping: float = 1.0e-6,
        max_joint_delta_per_step: float = 0.8,
        position_tolerance: float = 1.0e-3,
        orientation_tolerance: float = 2.0e-2,
        self_collision_check: bool = False,
    ):
        if self_collision_check:
            raise ValueError("M92UW evaluation requires Pink self_collision_check=false.")
        self.robot_origin_pose = self._validate_pose(robot_origin_pose, "robot_origin_pose")
        self.active_joints_name = [str(name) for name in active_joints_name]
        if not self.active_joints_name:
            raise ValueError("PinkPlanner requires at least one active joint.")
        self.dt = float(dt) if dt is not None else None
        if self.dt is None or not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("PinkPlanner dt must be a positive finite value.")

        frames = list(tool_frames or [])
        self.tool_frames = [str(frame) for frame in frames]
        if not self.tool_frames:
            raise ValueError("PinkPlanner requires tool_frames.")
        if len(set(self.tool_frames)) != len(self.tool_frames):
            raise ValueError(f"PinkPlanner tool frames must be unique: {frames}.")

        kinematics = _kinematics_from_yaml(yml_path)
        locked = _locked_joint_positions(kinematics)
        locked.update({str(name): float(value) for name, value in (locked_joint_positions or {}).items()})
        self._build_configuration(
            urdf_path,
            mesh_dir,
            locked,
        )

        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.frame_task_gain = float(frame_task_gain)
        self.frame_task_lm_damping = float(frame_task_lm_damping)
        self.posture_task_cost = float(posture_task_cost)
        self.null_space_task_cost = float(null_space_task_cost)
        self.previous_command_task_cost = float(previous_command_task_cost)
        self.previous_command_reset_threshold = float(previous_command_reset_threshold)
        self.damping_task_cost = float(damping_task_cost)
        self.qp_damping = float(qp_damping)
        self.max_joint_delta_per_step = float(max_joint_delta_per_step)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        if self.max_joint_delta_per_step <= 0.0:
            raise ValueError("max_joint_delta_per_step must be positive.")
        if (
            min(
                self.posture_task_cost,
                self.null_space_task_cost,
                self.previous_command_task_cost,
                self.damping_task_cost,
            )
            < 0.0
        ):
            raise ValueError("Pink regularization costs must be non-negative.")
        if self.previous_command_reset_threshold <= 0.0:
            raise ValueError("previous_command_reset_threshold must be positive.")

        self._configure_cspace_policy(
            kinematics.get("cspace", {}),
            null_space_weight_overrides,
        )

        self.frame_tasks = {
            frame: FrameTask(
                frame,
                position_cost=self.position_cost,
                orientation_cost=self.orientation_cost,
                lm_damping=self.frame_task_lm_damping,
                gain=self.frame_task_gain,
            )
            for frame in self.tool_frames
        }
        self._posture_task = PostureTask(cost=self.posture_task_cost)
        # Pink's task cost is a residual multiplier, so it is squared in the
        # QP objective. CuRobo's C-space vector multiplies the squared error
        # directly; sqrt preserves its per-joint objective ratios.
        self._posture_task.cost = self.posture_task_cost * np.sqrt(self.cspace_distance_weight)
        self._null_space_task = None
        if self.default_joint_position is not None:
            self._null_space_task = PostureTask(cost=self.null_space_task_cost)
            self._null_space_task.cost = self.null_space_task_cost * np.sqrt(self.null_space_weight)
            self._null_space_task.set_target(self.default_joint_position)
        self._previous_command_task = None
        self._previous_commands: dict[int, np.ndarray] = {}
        if self.previous_command_task_cost > 0.0:
            self._previous_command_task = PostureTask(cost=self.previous_command_task_cost)
            self._previous_command_task.cost = self.previous_command_task_cost * np.sqrt(self.cspace_distance_weight)
        self._damping_task = DampingTask(cost=self.damping_task_cost)
        self.solver = self._select_solver(solver)
        self._closed = False

    def _configure_cspace_policy(
        self,
        raw_cspace,
        null_space_weight_overrides: Mapping[str, float] | None,
    ) -> None:
        cspace = dict(raw_cspace or {})
        configured_names = [str(name) for name in cspace.get("joint_names", [])]

        def ordered_vector(key: str, default: float) -> np.ndarray:
            raw_values = cspace.get(key)
            if raw_values is None:
                return np.full(len(self.controlled_joint_names_pinocchio), default)
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            if len(configured_names) != values.size:
                raise ValueError(f"Pink cspace {key} has {values.size} values for {len(configured_names)} joint names.")
            by_name = dict(zip(configured_names, values, strict=True))
            missing = [name for name in self.controlled_joint_names_pinocchio if name not in by_name]
            if missing:
                raise ValueError(f"Pink cspace {key} is missing joints: {missing}.")
            ordered = np.asarray(
                [by_name[name] for name in self.controlled_joint_names_pinocchio],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(ordered)) or np.any(ordered < 0.0):
                raise ValueError(f"Pink cspace {key} must be finite and non-negative.")
            return ordered

        self.cspace_distance_weight = ordered_vector("cspace_distance_weight", 1.0)
        self.null_space_weight = ordered_vector("null_space_weight", 1.0)
        for name, raw_value in (null_space_weight_overrides or {}).items():
            name = str(name)
            if name not in self.controlled_joint_names_pinocchio:
                raise ValueError(f"Pink null-space override joint {name!r} is not controlled.")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Pink null-space override for {name!r} must be finite and non-negative.")
            index = self.controlled_joint_names_pinocchio.index(name)
            self.null_space_weight[index] = value
        raw_default = cspace.get("default_joint_position")
        if raw_default is None:
            raw_default = cspace.get("retract_config")
        if raw_default is None:
            self.default_joint_position = None
        else:
            values = np.asarray(raw_default, dtype=np.float64).reshape(-1)
            if len(configured_names) != values.size:
                raise ValueError(
                    "Pink cspace default_joint_position has "
                    f"{values.size} values for {len(configured_names)} joint names."
                )
            by_name = dict(zip(configured_names, values, strict=True))
            missing = [name for name in self.controlled_joint_names_pinocchio if name not in by_name]
            if missing:
                raise ValueError(f"Pink cspace default_joint_position is missing joints: {missing}.")
            self.default_joint_position = np.asarray(
                [by_name[name] for name in self.controlled_joint_names_pinocchio],
                dtype=np.float64,
            )

    def _build_configuration(
        self,
        urdf_path,
        mesh_dir,
        locked_positions,
    ):
        urdf_path = str(Path(urdf_path).expanduser().resolve())
        mesh_path = None if mesh_dir is None else str(Path(mesh_dir).expanduser().resolve())
        wrapper = (
            RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
            if mesh_path is not None
            else RobotWrapper.BuildFromURDF(urdf_path)
        )
        full_model = wrapper.model
        reference_q = np.asarray(wrapper.q0, dtype=np.float64).copy()
        all_joint_names = list(full_model.names[1:])
        missing = [name for name in self.active_joints_name if name not in all_joint_names]
        if missing:
            raise ValueError(f"Pink active joints are absent from the URDF: {missing}.")

        for name, value in locked_positions.items():
            joint_id = full_model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Pink locked joint {name!r} is absent from the URDF.")
            joint = full_model.joints[joint_id]
            angle_or_position = float(value)
            if not np.isfinite(angle_or_position):
                raise ValueError(f"Pink locked joint {name!r} has a non-finite value: {value!r}.")
            if joint.nq == 1 and joint.nv == 1:
                reference_q[joint.idx_q] = angle_or_position
            elif joint.nq == 2 and joint.nv == 1:
                # Pinocchio represents a continuous revolute joint on the unit
                # circle as [cos(q), sin(q)], despite it having one velocity DoF.
                reference_q[joint.idx_q : joint.idx_q + 2] = (
                    np.cos(angle_or_position),
                    np.sin(angle_or_position),
                )
            else:
                raise ValueError(
                    f"Pink locked joint {name!r} has nq={joint.nq}, nv={joint.nv}; "
                    "only scalar one-DoF joints are supported."
                )

        joints_to_lock = [
            full_model.getJointId(name) for name in all_joint_names if name not in self.active_joints_name
        ]
        model = pin.buildReducedModel(full_model, joints_to_lock, reference_q)
        data = model.createData()
        pin_joint_names = list(model.names[1:])
        if set(pin_joint_names) != set(self.active_joints_name):
            raise ValueError(
                "Pink reduced-model joints do not match requested joints: "
                f"model={pin_joint_names}, requested={self.active_joints_name}."
            )
        for name in pin_joint_names:
            joint = model.joints[model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"Pink controlled joint {name!r} must have nq=nv=1, got nq={joint.nq}, nv={joint.nv}.")

        reduced_q = np.asarray(
            [reference_q[full_model.joints[full_model.getJointId(name)].idx_q] for name in pin_joint_names],
            dtype=np.float64,
        )
        self.pink_configuration = Configuration(model, data, reduced_q)
        self.controlled_joint_names_pinocchio = pin_joint_names
        self._pin_to_active_order = np.asarray(
            [self.active_joints_name.index(name) for name in pin_joint_names],
            dtype=np.int64,
        )
        self._active_to_pin_order = np.asarray(
            [pin_joint_names.index(name) for name in self.active_joints_name],
            dtype=np.int64,
        )
        self.lower_limits = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        self.upper_limits = np.asarray(model.upperPositionLimit, dtype=np.float64)
        for frame in self.tool_frames:
            if not model.existFrame(frame):
                raise ValueError(f"Pink tool frame {frame!r} is absent from the URDF.")

    @staticmethod
    def _select_solver(requested: str | None) -> str:
        available = list(qpsolvers.available_solvers)
        if requested is not None:
            requested = str(requested).lower()
            if requested not in available:
                raise ValueError(f"Pink QP solver {requested!r} is unavailable; available={available}.")
            return requested
        for candidate in ("quadprog", "daqp", "osqp"):
            if candidate in available:
                return candidate
        if available:
            return available[0]
        raise RuntimeError("No QP solver is available for Pink.")

    @staticmethod
    def _validate_pose(value, label: str) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{label} must be a finite pose [x,y,z,qw,qx,qy,qz].")
        norm = float(np.linalg.norm(pose[3:7]))
        if norm <= 1.0e-12:
            raise ValueError(f"{label} contains a zero quaternion.")
        pose = pose.copy()
        pose[3:7] /= norm
        return pose

    def _active_to_pin(self, current_joint) -> np.ndarray:
        current = np.asarray(current_joint, dtype=np.float64).reshape(-1)
        if current.shape != (len(self.active_joints_name),):
            raise ValueError(f"Pink current q has shape {current.shape}; expected ({len(self.active_joints_name)},).")
        if not np.all(np.isfinite(current)):
            raise ValueError("Pink current q must contain only finite values.")
        return current[self._pin_to_active_order]

    @staticmethod
    def _world_to_base(base_pose, target_pose) -> pin.SE3:
        base = PinkPlanner._validate_pose(base_pose, "real_robot_pose")
        target = PinkPlanner._validate_pose(target_pose, "target pose")
        world_rotation_base = t3d.quaternions.quat2mat(base[3:7])
        world_rotation_target = t3d.quaternions.quat2mat(target[3:7])
        base_position_target = world_rotation_base.T @ (target[:3] - base[:3])
        base_rotation_target = world_rotation_base.T @ world_rotation_target
        return pin.SE3(base_rotation_target, base_position_target)

    def _normalize_targets(self, targets) -> dict[str, np.ndarray]:
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("Pink tool targets must be a non-empty frame-to-pose mapping.")
        unknown = [str(frame) for frame in targets if str(frame) not in self.frame_tasks]
        if unknown:
            raise ValueError(f"Unknown Pink tool frames {unknown}; configured={self.tool_frames}.")
        return {str(frame): self._validate_pose(pose, f"target[{frame}]") for frame, pose in targets.items()}

    def solve_differential_ik(
        self,
        curr_joint_pos,
        target_poses,
        real_robot_pose=None,
        *,
        env_idx=0,
    ):
        """Return one integrated joint-position command from measured qpos."""

        if self._closed:
            raise RuntimeError("PinkPlanner is closed.")
        root_pose = self.robot_origin_pose if real_robot_pose is None else real_robot_pose
        try:
            targets = self._normalize_targets(target_poses)
            current_pin = np.clip(
                self._active_to_pin(curr_joint_pos),
                self.lower_limits,
                self.upper_limits,
            )
            self.pink_configuration.update(current_pin)
            self._posture_task.set_target(current_pin)
            env_idx = int(env_idx)
            if self._previous_command_task is not None:
                previous_command = self._previous_commands.get(env_idx)
                if (
                    previous_command is None
                    or np.max(np.abs(previous_command - current_pin)) > self.previous_command_reset_threshold
                ):
                    previous_command = current_pin
                self._previous_command_task.set_target(previous_command)
            self._damping_task.cost = self.damping_task_cost

            tasks = []
            target_transforms = {}
            for frame, target_pose in targets.items():
                task = self.frame_tasks[frame]
                task.set_position_cost(self.position_cost)
                task.set_orientation_cost(self.orientation_cost)
                task.gain = self.frame_task_gain
                task.lm_damping = self.frame_task_lm_damping
                target = self._world_to_base(root_pose, target_pose)
                task.set_target(target)
                target_transforms[frame] = target
                tasks.append(task)
            tasks.append(self._posture_task)
            if self._previous_command_task is not None:
                tasks.append(self._previous_command_task)
            if self._null_space_task is not None:
                tasks.append(self._null_space_task)
            tasks.append(self._damping_task)

            step = self.max_joint_delta_per_step
            lower = np.maximum(self.lower_limits, current_pin - step)
            upper = np.minimum(self.upper_limits, current_pin + step)
            limits = [
                self.pink_configuration.model.configuration_limit,
                self.pink_configuration.model.velocity_limit,
                _JointPositionWindowLimit(lower, upper),
            ]
            velocity_pin = np.asarray(
                solve_ik(
                    self.pink_configuration,
                    tasks,
                    self.dt,
                    solver=self.solver,
                    damping=self.qp_damping,
                    limits=limits,
                    safety_break=False,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if velocity_pin.shape != current_pin.shape or not np.all(np.isfinite(velocity_pin)):
                raise ValueError("Pink returned an invalid joint velocity.")
            candidate_pin = pin.integrate(
                self.pink_configuration.model,
                current_pin,
                velocity_pin * self.dt,
            )
            candidate_pin = np.clip(candidate_pin, lower, upper)
            self.pink_configuration.update(candidate_pin)

            position_errors = []
            orientation_errors = []
            for frame, target in target_transforms.items():
                actual = self.pink_configuration.get_transform_frame_to_world(frame)
                position_errors.append(float(np.linalg.norm(actual.translation - target.translation)))
                delta_rotation = actual.rotation.T @ target.rotation
                cosine = np.clip((np.trace(delta_rotation) - 1.0) * 0.5, -1.0, 1.0)
                orientation_errors.append(float(np.arccos(cosine)))

            joint_value = candidate_pin[self._active_to_pin_order]
            joint_velocity = velocity_pin[self._active_to_pin_order]
            max_position_error = max(position_errors, default=0.0)
            max_orientation_error = max(orientation_errors, default=0.0)
            if self._previous_command_task is not None:
                self._previous_commands[env_idx] = candidate_pin.copy()
            return {
                "status": "Success",
                "joint_names": list(self.active_joints_name),
                "joint_value": joint_value,
                "joint_velocity": joint_velocity,
                "joint_acceleration": np.zeros_like(joint_velocity),
                "tracked_tool_frames": list(targets),
                "position_error": max_position_error,
                "orientation_error": max_orientation_error,
                "self_collision_check": False,
                "minimum_self_collision_distance": None,
                "self_collision_clear": True,
                "converged": (
                    max_position_error <= self.position_tolerance
                    and max_orientation_error <= self.orientation_tolerance
                ),
                "solver": self.solver,
            }
        except Exception as exc:
            return {
                "status": "Fail",
                "joint_names": list(self.active_joints_name),
                "joint_value": np.asarray(curr_joint_pos, dtype=np.float64).copy(),
                "tracked_tool_frames": list(target_poses) if isinstance(target_poses, Mapping) else [],
                "error": f"{type(exc).__name__}: {exc}",
                "solver": self.solver,
                "self_collision_check": False,
            }

    def reset(self, current_joint_pos=None, *, env_idx=None):
        """Clear all command histories, or reset one environment's reference."""

        if current_joint_pos is None:
            if env_idx is None:
                self._previous_commands.clear()
            else:
                self._previous_commands.pop(int(env_idx), None)
        else:
            current_pin = np.clip(
                self._active_to_pin(current_joint_pos),
                self.lower_limits,
                self.upper_limits,
            )
            self._previous_commands[0 if env_idx is None else int(env_idx)] = current_pin.copy()

    def close(self):
        self.reset()
        self._closed = True
