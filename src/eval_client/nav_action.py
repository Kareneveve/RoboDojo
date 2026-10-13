"""Translate navigation policy actions into whole-body control frames.

The adapter owns action decoding, IK targets and joint/chassis conversion.
The environment supplies the control interval and schedules the resulting frame.
"""

from __future__ import annotations

import logging

import numpy as np

from env.robot_manager.control_manager import MetaControl
from utils.transformer import matrix_to_pose, pose_to_matrix

logger = logging.getLogger(__name__)


class NavActionAdapter:
    """Resolve policy actions using the robot manager's measured state.

    Only the robot manager is retained; the caller supplies the current control
    interval for each frame so changes in collection timing take effect at once.
    """

    LINEAR_VELOCITY_KEY = "root_linear_velocity"
    ANGULAR_VELOCITY_KEY = "root_angular_velocity"

    def __init__(
        self,
        robot_manager,
        *,
        linear_velocity_key=LINEAR_VELOCITY_KEY,
        angular_velocity_key=ANGULAR_VELOCITY_KEY,
    ):
        self.robot_manager = robot_manager
        self.LINEAR_VELOCITY_KEY = linear_velocity_key
        self.ANGULAR_VELOCITY_KEY = angular_velocity_key

    def get_action_type(self, action):
        """Return ``"joint"``, ``"ee"``, or None for chassis-only actions.

        Recognizes world-frame and base_link-frame end-effector keys,
        and lets an action carry nothing but a chassis velocity so a
        pure navigation policy can drive the robot without commanding the
        upper body.
        """
        action_type = []
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            arm_key = self.robot_manager.process_name(robot.arm_name)
            if arm_key in action and "joint" not in action_type:
                action_type.append("joint")
            if "ee" in action_type:
                continue
            if any(key in action for key in self._ee_pose_keys(robot)):
                action_type.append("ee")
        if len(action_type) > 1:
            logger.warning(
                "Multiple action types %s found in action keys %s; using %s.",
                action_type,
                list(action),
                action_type[0],
            )
            return action_type[0]
        if not action_type:
            if self._base_velocity_command(action) is None:
                raise ValueError(f"Cannot infer action type from action dict keys: {action.keys()}")
            return None
        return action_type[0]

    def build_control_info(self, action, action_type, env_idx, *, interval_s):
        """Resolve one action into a single qpos/qvel + chassis frame.

        ``interval_s`` is the duration of the environment's control interval
        in seconds, used to compute qvel from the measured joint positions.
        """

        if action_type == "joint":
            groups = self._joint_action_groups(action)
        elif action_type == "ee":
            groups = self._ee_action_groups(action, env_idx)
        else:
            groups = []

        qpos, qvel = self._qpos_qvel_from_groups(groups, env_idx, interval_s)
        control_info = {"qpos": qpos, "qvel": qvel}

        velocity = self._base_velocity_command(action)
        yaw_joint_name = self._yaw_velocity_joint_name()
        if velocity is not None:
            control_info[MetaControl.BASE_CONTROL_KEY] = {
                "control_mode": "direct_velocity",
                "velocity": velocity,
            }
        # RobotManager.control_robot folds the yaw joint's velocity target
        # into the base command, so a stale cached target would override
        # the commanded yaw rate (and keep an uncommanded chassis turning).
        if yaw_joint_name is not None:
            qvel[yaw_joint_name] = 0.0 if velocity is None else float(velocity[5])
        return control_info

    def _qpos_qvel_from_groups(self, groups, env_idx, interval_s):
        """Named qpos targets plus the qvel that reaches them this interval.

        ``qvel`` is ``(q_target - q_measured) / interval_s``, measured once
        at the start of the interval, so a lagging joint is commanded to
        move faster and an overshooting one to slow down.
        """
        qpos: dict[str, float] = {}
        qvel: dict[str, float] = {}
        for robot, joint_names, values in groups:
            joint_names = list(joint_names)
            if not joint_names:
                continue
            measured = self.robot_manager.get_named_joint_positions(joint_names, env_idx, robot=robot)
            action_names = self.robot_manager._canonical_action_joint_names(joint_names, robot)
            for name, target, current in zip(action_names, values, measured, strict=True):
                target = float(target)
                qpos[name] = target
                qvel[name] = (target - float(current)) / interval_s
        return qpos, qvel

    def _joint_action_groups(self, action):
        """Arm, end-effector and waist/leg targets from a joint action."""

        groups = []
        shared_articulations: set[int] = set()
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            arm_key = self.robot_manager.process_name(robot.arm_name)
            if arm_key in action:
                groups.append(
                    (
                        robot,
                        list(robot.arm_joints_name),
                        self._group_values(action[arm_key], robot.arm_joints_name, arm_key),
                    )
                )
            groups.extend(self._end_effector_groups(robot, action))

            articulation = self.robot_manager.robot_key[self.robot_manager.robot_list.index(robot)]
            if id(articulation) in shared_articulations:
                continue
            shared_articulations.add(id(articulation))
            for state_key, names_attr in (
                ("waist_joint_state", "waist_joints_name"),
                ("leg_joint_state", "leg_joints_name"),
            ):
                joint_names = list(getattr(robot, names_attr, []) or [])
                if state_key not in action or not joint_names:
                    continue
                groups.append(
                    (
                        robot,
                        joint_names,
                        self._group_values(action[state_key], joint_names, state_key),
                    )
                )
        return groups

    def _ee_action_groups(self, action, env_idx):
        """Coupled upper-body IK output plus the commanded hand joints.

        The IK control group already owns the waist and leg columns, so an
        explicit waist/leg command is ignored in ee mode.
        """
        groups = []
        ik_group = self._solve_upper_body_ik(action, env_idx)
        if ik_group is not None:
            groups.append(ik_group)
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            groups.extend(self._end_effector_groups(robot, action))
        return groups

    def _end_effector_groups(self, robot, action):
        gripper_key = self.robot_manager.process_name(robot.gripper_name)
        if gripper_key not in action:
            return []
        if robot.ee_type != "hand":
            raise ValueError("NavEval currently supports hand joint targets only.")
        joint_names = list(robot.gripper_joints_name)
        values = self._group_values(action[gripper_key], joint_names, gripper_key)
        return [(robot, joint_names, values)]

    @staticmethod
    def _group_values(value, joint_names, key):
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size != len(joint_names):
            raise ValueError(
                f"action['{key}'] has {values.size} values for {len(joint_names)} joints {list(joint_names)}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"action['{key}'] must be finite.")
        return values

    def _solve_upper_body_ik(self, action, env_idx):
        """Solve every arm's tool frame at once, plus head when provided."""

        targets = {}
        for robot in self.robot_manager.robot_list:
            if robot.type != "target":
                continue
            pose = self._read_tool_target(action, robot, env_idx)
            if pose is None:
                raise ValueError(
                    "Action type 'ee' requires an end-effector pose for "
                    f"{robot.arm_name}; expected one of "
                    f"{self._ee_pose_keys(robot)}, got {sorted(action)}."
                )
            frame = getattr(robot, "target_tool_frame", robot.ee_link_name)
            targets[frame] = pose

        head_frame, head_pose = self._read_head_target(action, env_idx)
        if head_pose is not None:
            targets[head_frame] = head_pose

        planner = self.robot_manager.get_planner(None)
        result = self.robot_manager.solve_tools(
            targets,
            env_idx,
            trans="world",
        )
        if result is None or result.get("status") != "Success":
            logger.warning(
                "env%s IK failed for tool frames %s; holding the previous joint targets.",
                env_idx,
                sorted(targets),
            )
            return None
        return (
            planner.owner,
            list(result["joint_names"]),
            np.asarray(result["joint_value"], dtype=np.float64).reshape(-1),
        )

    def _ee_pose_keys(self, robot):
        """World-frame and base_link-frame action keys for one arm."""

        if self.robot_manager.target_arm_nums == 1:
            return ("ee_pose", "base_ee_pose")
        side = robot.arm_name.split("_")[0]
        return (f"{side}_ee_pose", f"{side}_base_ee_pose")

    def _read_tool_target(self, action, robot, env_idx):
        world_key, base_key = self._ee_pose_keys(robot)
        if world_key in action:
            return np.asarray(action[world_key], dtype=np.float64).reshape(7)
        if base_key in action:
            return self._base_pose_to_world(action[base_key], env_idx)
        return None

    def _read_head_target(self, action, env_idx):
        """Return the frame a head command steers, with the pose for it."""

        if "head_base_pose" not in action:
            return None, None
        pose = self._base_pose_to_world(action["head_base_pose"], env_idx)
        # Raises when the robot names a head frame the planner does not
        # constrain, which would drop the head command silently.
        head_frame = self.robot_manager.head_tool_frame()
        if head_frame is None:
            raise ValueError(
                "A head target requires the robot to declare common.head_tool_frame in its robot_config.yml."
            )
        return head_frame, pose

    def _base_pose_to_world(self, pose, env_idx):
        """Lift a base_link-frame pose into the env-relative world frame.

        The planner takes env-relative targets and rebases them itself, so
        a base_link-frame command has to be composed with the measured
        base_link pose first.
        """
        base_pose = self.robot_manager.get_base_link_pose(
            env_idx_list=[env_idx],
            is_relative=True,
        )[env_idx]
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        return matrix_to_pose(pose_to_matrix(base_pose) @ pose_to_matrix(pose))

    def _base_velocity_command(self, action):
        """Chassis command as a 6D world velocity, or None when absent.

        The chassis can only realize planar motion, so vertical linear
        velocity and the roll/pitch rates are dropped.
        """
        linear = action.get(self.LINEAR_VELOCITY_KEY)
        angular = action.get(self.ANGULAR_VELOCITY_KEY)
        if linear is None and angular is None:
            return None
        velocity = np.zeros(6, dtype=np.float64)
        if linear is not None:
            velocity[:2] = np.asarray(linear, dtype=np.float64).reshape(-1)[:2]
        if angular is not None:
            velocity[5] = float(np.asarray(angular, dtype=np.float64).reshape(-1)[2])
        if not np.all(np.isfinite(velocity)):
            raise ValueError(f"Chassis velocity command must be finite, got {velocity}.")
        return velocity

    def _yaw_velocity_joint_name(self):
        adapter = getattr(self.robot_manager, "mobile_base_adapter", None)
        if adapter is None:
            return None
        name = getattr(adapter, "yaw_velocity_joint_name", None)
        return None if name is None else str(name)
