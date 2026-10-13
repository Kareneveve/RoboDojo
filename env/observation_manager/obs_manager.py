from copy import deepcopy
from typing import List

import numpy as np

from env.description_manager.desc_manager import DescManager
from env.environment.isaac.physx_device import as_numpy


class ObsManager:
    ANNOTATORS_TO_COLLECT = {
        "rgb": "color",
        "depth": "depth",
        "distance_to_image_plane": "distance_to_image_plane",
    }

    def __init__(
        self,
        obs_config,
        num_envs,
        dt,
        task_name,
        description_cfg,
        seeds_per_env: List[int] = None,
    ):
        self.obs_config = obs_config
        self.num_envs = num_envs
        self.dt = dt
        self.task_name = task_name
        self.description_cfg = description_cfg
        self.desc_manager = DescManager(
            num_envs=self.num_envs,
            description_cfg=self.description_cfg,
            desc_type="seen",
            seeds_per_env=seeds_per_env,
        )
        self.robot_cfg = obs_config.get("robot", {})
        self.collect_base_ee_pose = bool(self.robot_cfg.get("base_ee_pose", False))
        self.collect_base_head_pose = bool(self.robot_cfg.get("head_base_pose", False))
        self.collect_root_velocity = bool(self.robot_cfg.get("root_velocity", False))
        self.collect_freq = self.obs_config.get("collect_freq", 0)
        # Frames the SDG dispatcher must deliver after the physics step before
        # images are read. One is exact with the zero-delay Kit settings that
        # the entrypoints enable; render_sync adds one more if they are absent.
        self.capture_render_passes = int(self.obs_config.get("capture_render_passes", 1))
        self.capture_render_max_passes = int(self.obs_config.get("capture_render_max_passes", 16))
        if self.collect_freq > 0:
            self.collect_interval = 1.0 / (self.dt * self.collect_freq)
        else:
            self.collect_interval = 0.0
        if not self.collect_interval.is_integer():
            raise ValueError("Collect Interval must be integer!")
        vision_cfg = self.obs_config.get("vision", {})
        self.collect_approximate_depth = vision_cfg.get("approximate_depth", True)
        self.collect_depth = vision_cfg.get("depth", False)
        self.collect_intrinsic_matrix = vision_cfg.get("intrinsic_matrix", False)
        self.collect_extrinsic_matrix = vision_cfg.get("extrinsic_matrix", False)
        self.collect_extrinsic_base_matrix = vision_cfg.get("extrinsic_base_matrix", False)
        self.collect_distortion = vision_cfg.get("distortion", False)
        self.collect_shape = vision_cfg.get("shape", True)
        self.extrinsics_in_env_frame = False

    def initialize(self, env):
        self.env = env
        if hasattr(env, "robot_manager"):
            self.robot_manager = env.robot_manager
        else:
            self.robot_manager = None
        if hasattr(env, "capture_manager"):
            self.capture_manager = env.capture_manager
        else:
            self.capture_manager = None
        if hasattr(env, "camera_manager"):
            self.camera_manager = env.camera_manager
        else:
            self.camera_manager = None

        self.initialize_instruction()

    def initialize_instruction(self):
        self.desc_manager.initialize(self.env)
        self.instruction = self.desc_manager.get_one_description()

    def reset(self):
        self.desc_manager.reset()
        self.instruction = self.desc_manager.get_one_description()

    def _collect_shared_joint_groups(self, obs, env_idx_list, robot):
        for name, indices, getter in (
            ("waist_joint_state", getattr(robot, "waist_joint_indices", []), self.robot_manager.get_waist_joint),
            ("leg_joint_state", getattr(robot, "leg_joint_indices", []), self.robot_manager.get_leg_joint),
        ):
            if not len(indices):
                continue
            joints = getter(robot=robot, env_idx_list=env_idx_list)
            for env_idx in env_idx_list:
                obs[env_idx]["state"][name] = joints[env_idx]
                obs[env_idx]["action"][name] = joints[env_idx]

    def _world_to_base_matrices(self, env_idx_list):
        """Map CameraManager's world frame into the measured robot base_link."""
        if not self.collect_extrinsic_base_matrix or self.robot_manager is None:
            return {}
        from utils.transformer import pose_to_matrix

        base_poses = self.robot_manager.get_base_link_pose(env_idx_list=env_idx_list, is_relative=False)
        return {
            env_idx: np.linalg.inv(pose_to_matrix(as_numpy(base_poses[env_idx])))
            for env_idx in env_idx_list
            if base_poses.get(env_idx) is not None
        }

    def render_for_capture(self):
        """Render until the camera buffers hold the current physics state.

        Call this instead of a bare ``env.render()`` right before ``get_obs``.
        """
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(
            self.env,
            capture_manager=self.capture_manager,
            min_passes=self.capture_render_passes,
            max_passes=self.capture_render_max_passes,
        )

    def get_obs(self, env_idx_list=None):  # batch
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)
        obs = dict()
        for env_idx in env_idx_list:
            obs[env_idx] = dict()
            obs[env_idx]["vision"] = dict()
            obs[env_idx]["state"] = dict()
            obs[env_idx]["action"] = dict()
            obs[env_idx]["data_format_version"] = "v1.0"
            obs[env_idx]["additional_info"] = {
                "frequency": self.collect_freq,
            }
            obs[env_idx]["instruction"] = self.instruction[env_idx]
        if self.camera_manager is not None and self.capture_manager is not None:
            try:
                data = self.capture_manager.step(env_ids=env_idx_list)
                for ith in range(self.camera_manager.num_cams):
                    cam_data = data[ith]
                    for annotator_name, env_list in cam_data.items():
                        collect_name = self.ANNOTATORS_TO_COLLECT[annotator_name]
                        for idx, env_idx in enumerate(env_idx_list):
                            camera_name = self.camera_manager.camera_names[env_idx][ith]
                            if camera_name not in obs[env_idx]["vision"]:
                                obs[env_idx]["vision"][camera_name] = dict()
                            obs[env_idx]["vision"][camera_name][collect_name] = env_list[idx]["data"]

                            if annotator_name == "rgb":
                                obs[env_idx]["vision"][camera_name][collect_name] = obs[env_idx]["vision"][camera_name][
                                    collect_name
                                ][:, :, :3]
                                if self.collect_shape:
                                    obs[env_idx]["vision"][camera_name]["shape"] = obs[env_idx]["vision"][camera_name][
                                        collect_name
                                    ].shape

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Camera observation capture failed.")
                raise e

            for env_idx in env_idx_list:
                for cam_data in obs[env_idx]["vision"].values():
                    if "distance_to_image_plane" in cam_data:
                        raw = cam_data.pop("distance_to_image_plane")
                        if hasattr(raw, "ndim") and raw.ndim >= 1 and raw.shape[-1] == 1:
                            raw = raw.squeeze(-1)
                        if self.collect_approximate_depth:
                            cam_data["approximate_depth"] = np.clip(raw * 1000, 0, 65535).astype(np.uint16)
                        if self.collect_depth:
                            cam_data["depth"] = raw.astype(np.float32)

            if (
                self.collect_intrinsic_matrix
                or self.collect_extrinsic_matrix
                or self.collect_extrinsic_base_matrix
                or self.collect_distortion
            ):
                world_to_base = self._world_to_base_matrices(env_idx_list)
                for ith in range(self.camera_manager.num_cams):
                    for idx, env_idx in enumerate(env_idx_list):
                        camera_name = self.camera_manager.camera_names[env_idx][ith]
                        camera_obs = obs[env_idx]["vision"].get(camera_name)
                        if camera_obs is None:
                            continue
                        if self.collect_intrinsic_matrix:
                            camera_obs["intrinsic_matrix"] = self.camera_manager.get_camera_intrinsics(ith, env_idx)
                        if self.collect_extrinsic_matrix or env_idx in world_to_base:
                            extrinsic = self.camera_manager.get_camera_extrinsics(ith, env_idx).copy()
                            if env_idx in world_to_base:
                                camera_obs["extrinsic_base_matrix"] = world_to_base[env_idx] @ extrinsic
                            if self.collect_extrinsic_matrix:
                                if self.extrinsics_in_env_frame:
                                    extrinsic[:3, 3] -= as_numpy(self.env.sim.scene.env_origins[env_idx])
                                camera_obs["extrinsic_matrix"] = extrinsic
                        if self.collect_distortion:
                            camera_obs.update(self.camera_manager.get_camera_distortion(ith, env_idx))

        if self.robot_manager is not None:
            try:
                collected_body_articulations = set()
                for robot_idx, robot in enumerate(self.robot_manager.robot_list):
                    if robot.type != "target":
                        continue
                    if self.robot_cfg["joint_states"]:
                        if robot.robot_type == "arm":
                            joints = self.robot_manager.get_joint(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if robot.robot_type == "arm":
                                obs[env_idx]["state"][f"{robot.arm_name}_joint_state"] = joints[env_idx]
                                obs[env_idx]["action"][f"{robot.arm_name}_joint_state"] = joints[env_idx]

                        articulation_id = id(self.robot_manager.robot_key[robot_idx])
                        if articulation_id not in collected_body_articulations:
                            self._collect_shared_joint_groups(obs, env_idx_list, robot)
                            collected_body_articulations.add(articulation_id)

                    if self.robot_cfg.get("world_ee_state", False):
                        endpose = self.robot_manager.get_real_endpose(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if self.robot_manager.target_arm_nums == 1:
                                obs[env_idx]["state"]["ee_pose"] = endpose[env_idx]
                            else:
                                name = robot.arm_name.split("_")[0]
                                obs[env_idx]["state"][f"{name}_ee_pose"] = endpose[env_idx]

                    if self.collect_base_ee_pose:
                        poses = self.robot_manager.get_base_endpose(robot=robot, env_idx_list=env_idx_list)
                        key = (
                            "base_ee_pose"
                            if self.robot_manager.target_arm_nums == 1
                            else f"{robot.arm_name.split('_')[0]}_base_ee_pose"
                        )
                        for env_idx in env_idx_list:
                            if poses[env_idx] is not None:
                                obs[env_idx]["state"][key] = np.asarray(poses[env_idx], dtype=np.float64).copy()

                if self.collect_base_head_pose:
                    poses = self.robot_manager.get_base_head_pose(env_idx_list=env_idx_list)
                    for env_idx in env_idx_list:
                        if poses[env_idx] is not None:
                            obs[env_idx]["state"]["head_base_pose"] = np.asarray(
                                poses[env_idx], dtype=np.float64
                            ).copy()

                if hasattr(self.robot_manager, "control_manager"):
                    for env_idx in env_idx_list:
                        for (
                            key,
                            data,
                        ) in self.robot_manager.control_manager.prev_control[env_idx].items():
                            position = data["position"]
                            if key.endswith("ee_joint_state"):
                                end_effector_name = self.robot_manager.restore_name(deepcopy(key))
                                robot = self.robot_manager.get_robot_by_gripper_name(end_effector_name)
                                if robot.ee_type == "gripper":
                                    val = position[0]
                                    if robot.gripper_move["sign"] == 1:
                                        val = (val - robot.gripper_scale[0]) / (
                                            robot.gripper_scale[1] - robot.gripper_scale[0]
                                        )
                                    else:
                                        val = (robot.gripper_scale[1] - val) / (
                                            robot.gripper_scale[1] - robot.gripper_scale[0]
                                        )
                                    obs[env_idx]["action"][key] = [val]
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = [val]
                                else:
                                    obs[env_idx]["action"][key] = position
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = position

                if self.collect_root_velocity:
                    measured = self.robot_manager.get_root_velocity(env_idx_list=env_idx_list)
                    adapter = self.robot_manager.mobile_base_adapter
                    for env_idx in env_idx_list:
                        commanded = self.robot_manager.get_last_applied_root_velocity(env_idx)
                        commanded = np.zeros(6) if commanded is None else np.asarray(commanded, dtype=np.float64)
                        if adapter is not None:
                            targets = adapter.get_joint_velocity_targets(env_idx)
                            commanded[5] = targets[adapter.yaw_velocity_joint_name]
                        state_velocity = np.asarray(measured[env_idx], dtype=np.float64)
                        obs[env_idx]["action"]["root_linear_velocity"] = commanded[:3].copy()
                        obs[env_idx]["action"]["root_angular_velocity"] = commanded[3:].copy()
                        obs[env_idx]["state"]["root_linear_velocity"] = state_velocity[:3].copy()
                        obs[env_idx]["state"]["root_angular_velocity"] = state_velocity[3:].copy()

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Robot observation capture failed.")
                raise e
        return obs
