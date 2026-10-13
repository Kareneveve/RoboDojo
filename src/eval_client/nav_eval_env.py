"""MiVerse evaluation lifecycle and whole-body control scheduling.

Resolve qpos/qvel once from measured state and hold the frame throughout the
collection interval. Chassis velocities use the direct_velocity channel.
"""

from __future__ import annotations

from copy import deepcopy
from functools import cache
import logging

import numpy as np

from env.robot_manager.control_manager import MetaControl
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.stability import StabilityLimits
from src.eval_client.eval_env import (
    build_eval_env_class,
    resolve_task_class,
)
from src.eval_client.nav_action import NavActionAdapter
from utils.cluttered_generator import UnStableError

logger = logging.getLogger(__name__)


def _validate_nav_entry(key, value, expected_dim):
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError(f"action_dict[{key!r}] must be array-like, got {type(value)}")
    array = np.asarray(value)
    if array.shape != (expected_dim,):
        raise ValueError(f"action_dict[{key!r}] must have shape ({expected_dim},), got {array.shape}")


@cache
def build_nav_eval_env_class(task_class):
    """Build the NavEvalEnv class on top of the dynamically built EvalEnv.

    EvalEnv itself is generated at runtime from the task class, so the
    navigation variant has to be built through the same indirection instead of
    a plain module-level `class NavEvalEnv(EvalEnv)`.
    """
    eval_env_class = build_eval_env_class(task_class)

    class NavEvalEnv(eval_env_class):
        # Shared (non-arm) joint groups that ObsManager emits for humanoid
        # embodiments, mapped to their dim entry in env_cfg/robot/_robot_info.json.
        NAV_JOINT_GROUPS = {
            "waist_joint_state": "waist_dim",
            "leg_joint_state": "leg_dim",
        }
        # Extra fixed-width action channels a navigation policy may send.
        NAV_STATE_DIMS = {
            "head_base_pose": 7,
            "base_ee_pose": 7,
            "left_base_ee_pose": 7,
            "right_base_ee_pose": 7,
            "root_linear_velocity": 3,
            "root_angular_velocity": 3,
        }
        LINEAR_VELOCITY_KEY = NavActionAdapter.LINEAR_VELOCITY_KEY
        ANGULAR_VELOCITY_KEY = NavActionAdapter.ANGULAR_VELOCITY_KEY

        def __init__(self, config, app, resume_state=None, **kwargs):
            config = deepcopy(config)
            room_name = config.scene.get("Room", {}).get("default")
            self._authored_base_pose = RobotManager._apply_room_robot_pose(config, room_name)
            super().__init__(config, app, resume_state=resume_state, **kwargs)
            self.obs_manager.extrinsics_in_env_frame = True
            robot_manager = getattr(self, "robot_manager", None)
            if robot_manager is not None:
                self._nav_action_adapter = NavActionAdapter(
                    robot_manager,
                    linear_velocity_key=self.LINEAR_VELOCITY_KEY,
                    angular_velocity_key=self.ANGULAR_VELOCITY_KEY,
                )

        def reset(self, seed=None, options=None):
            """Restore authored joints/base after settling, then warm up cameras."""
            super().reset(seed=seed, options=options)
            self.robot_manager.write_robot_init_joint_state_to_sim()
            self.reset_base_to_authored_pose()
            self.sim_step(render=False)
            # Hand observations use prev_control; sync it after the final reset pose.
            self.robot_manager.set_robot_init_state()
            for planner in self.robot_manager.planner.values():
                if not callable(getattr(planner, "reset", None)):
                    continue
                for env_idx in range(self.num_envs):
                    planner.reset(
                        current_qpos=self.robot_manager.get_named_joint_positions(
                            planner.joints, env_idx, planner.owner
                        ),
                        env_idx=env_idx,
                    )
            for _ in range(10):
                self.render()
            self.obs_manager.render_for_capture()

        def reset_base_to_authored_pose(self):
            if self._authored_base_pose is not None:
                for env_idx in range(self.num_envs):
                    self.robot_manager.teleport_root_pose(self._authored_base_pose, env_idx=env_idx)

        def setup_scene(self):
            self.scene_manager.apply_saved_poses(env_idx_list=list(range(self.num_envs)))
            self._align_layout_success()
            success, unstable_envs = self.scene_manager.layout_manager.check_layout_stability(
                self,
                render=False,
                maxstep=120,
                window=40,
                limits=StabilityLimits(
                    angle_deg=5.0,
                    horizontal_m=0.01,
                    downward_m=0.04,
                    upward_m=0.005,
                ),
                window_limits=StabilityLimits(
                    angle_deg=5.0,
                    horizontal_m=0.01,
                    downward_m=0.04,
                    upward_m=0.005,
                ),
            )
            unstable_envs = [idx for idx in set(unstable_envs) if idx < self.num_envs]
            self.unstable_nums += len(unstable_envs)
            self.episode_nums -= len(unstable_envs)
            if not success or self.episode_nums <= 0:
                raise UnStableError("All scene Unstable Error!")
            for _ in range(10):
                self.render()
            for idx in range(200):
                self.sim_step()
                if idx % 5 == 0:
                    self.obs_manager.render_for_capture()
                    self.obs_manager.get_obs()
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()

        def validate_action_dict(self, state_dict: dict) -> None:
            """Validate arm/ee keys as usual, plus the navigation channels.

            Waist, leg, head and chassis entries are per-articulation rather
            than per-arm, so they carry no left_/right_ prefix and are absent
            from the base validator's expected key set. They are checked here
            and stripped out before delegating, since the base validator
            rejects unknown keys.
            """
            for key, dim_key in self.NAV_JOINT_GROUPS.items():
                if key not in state_dict:
                    continue
                dims = self.robot_action_dim_info.get(dim_key)
                if not dims:
                    raise ValueError(
                        f"state_dict contains '{key}' but the configured robot "
                        f"declares no '{dim_key}' in _robot_info.json"
                    )
                if isinstance(dims, (list, tuple)):
                    expected_dim = int(sum(dims))
                else:
                    expected_dim = int(dims)
                _validate_nav_entry(key, state_dict[key], expected_dim)

            for key, expected_dim in self.NAV_STATE_DIMS.items():
                if key in state_dict:
                    _validate_nav_entry(key, state_dict[key], expected_dim)

            nav_keys = {*self.NAV_JOINT_GROUPS, *self.NAV_STATE_DIMS}
            super().validate_action_dict({k: v for k, v in state_dict.items() if k not in nav_keys})

        def get_action_type(self, action):
            """Return ``"joint"``, ``"ee"``, or None for chassis-only actions."""
            return self._nav_action_adapter.get_action_type(action)

        def take_action_batch(self, actions_list, env_idx_list=None):
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))

            control_env_idx_list = []
            control_info_list = []
            for idx, env_idx in enumerate(env_idx_list):
                action = actions_list[idx]
                self.validate_action_dict(action)
                action_type = self.get_action_type(action)
                if self.take_action_cnt[env_idx] == self.step_lim or self.end_flag[env_idx]:
                    continue
                self.take_action_cnt[env_idx] += 1
                logger.debug("env%s step: %s / %s", env_idx, self.take_action_cnt[env_idx], self.step_lim)
                control_info = self._build_control_info(action, action_type, env_idx)
                control_env_idx_list.append(env_idx)
                control_info_list.append(self.process_control_info(control_info, env_idx))

            if len(control_info_list) > 0:
                self.robot_manager.control_manager.push(control_env_idx_list, control_info_list)
                while not self.have_empty(control_env_idx_list):
                    self.step(env_idx_list=control_env_idx_list)
                    if self.physx_monitor_enabled:
                        self._check_physx_broken_envs()
                        self._check_endpose_finite(control_env_idx_list)

            self.reward_manager.step(env_idx_list=env_idx_list)
            self.is_episode_end()

        def process_control_info(self, control_info, env_idx):
            """Hold one already-resolved target frame for the whole interval.

            The qpos target and its closed-loop qvel are computed once per
            collect interval in ``_build_control_info``, so every physics step
            of the interval writes the same command. This replaces the base
            class' 80% interpolate / 20% hold ramp.

            The same ``MetaControl`` is queued for every physics step so
            ``get_action`` and the pop-merge cache hit on the second step.
            """
            del env_idx
            meta = control_info if isinstance(control_info, MetaControl) else MetaControl(control_info)
            return [meta] * self._control_steps()

        def _control_steps(self) -> int:
            """Physics steps per collect interval (``1 / (dt * collect_freq)``)."""

            return max(1, int(round(float(self.obs_manager.collect_interval))))

        def _control_interval_s(self) -> float:
            return self._control_steps() * float(self.dt)

        def _build_control_info(self, action, action_type, env_idx):
            """Resolve a control frame for the current collection interval."""
            return self._nav_action_adapter.build_control_info(
                action, action_type, env_idx, interval_s=self._control_interval_s()
            )

    return NavEvalEnv


def create_nav_eval_env(config, app, resume_state=None, **kwargs):
    nav_eval_env_class = build_nav_eval_env_class(resolve_task_class(config))
    return nav_eval_env_class(config, app, resume_state=resume_state, **kwargs)
