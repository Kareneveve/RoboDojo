from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path

from client_server.ws.model_client import WsModelClient
import numpy as np
import transforms3d as t3d
import websockets

from env.global_configs import *
from env.global_configs import BENCHMARK
from env.observation_manager.obs_manager import ObsManager
from env.seed_manager.seed_manager import SeedManager
from src.eval_client.path_utils import (
    make_run_id,
    safe_path_component,
    validate_resume_save_dir,
)
from utils.pipeline_utils import get_robot_action_dim_info
from utils.save_file import VideoStreamWriter, format_video_saved_message, save_json


def _patch_websockets_proxy_compat():
    # Isaac Sim may load an older bundled websockets package that does not
    # accept the proxy kwarg used by XPolicyLab's websocket client.
    connect = websockets.connect
    if getattr(connect, "_robodojo_proxy_compat", False):
        return

    try:
        if "proxy" in inspect.signature(connect).parameters:
            return
    except (TypeError, ValueError):
        pass

    def connect_without_proxy(*args, proxy=None, **kwargs):
        _ = proxy
        return connect(*args, **kwargs)

    connect_without_proxy._robodojo_proxy_compat = True
    websockets.connect = connect_without_proxy


def create_eval_env(config, app, resume_state=None, **kwargs):
    task_name = config.eval_cfg.get("task_name", None)
    if task_name is None:
        raise ValueError("Task name must be specified in eval_cfg!")

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_class = task_registry.load_task_class(task_name)
    config.eval_cfg["task_name"] = task_name

    class EvalEnv(task_class):
        def __init__(self, config, app, resume_state=None, **kwargs):
            super().__init__(config, app, **kwargs)
            self.eval_cfg = config.eval_cfg
            self.config_name = self.eval_cfg.get("config_name", None)
            self.task_name = self.eval_cfg.get("task_name", None)
            self.eval_batch = self.eval_cfg.get("eval_batch", False)
            self.eval_num = int(self.eval_cfg.get("eval_num", 50))
            self.policy_name = self.eval_cfg.get("policy_name", None)
            self.additional_info = self.eval_cfg.get("additional_info", "")
            self.eval_seed = self.eval_cfg.get("seed", 0)
            self.physx_monitor_enabled = bool(self.eval_cfg.get("physx_monitor_enabled", False))
            if self.physx_monitor_enabled:
                from src.eval_client.physx_warning_monitor import (
                    PhysXBrokenError,
                    PhysXFatalError,
                    get_monitor,
                )

                self._physx_get_monitor = get_monitor
                self._PhysXBrokenError = PhysXBrokenError
                self._PhysXFatalError = PhysXFatalError

            run_id = os.environ.get("ROBODOJO_RUN_ID")
            if not run_id:
                run_id = make_run_id()
                os.environ["ROBODOJO_RUN_ID"] = run_id
            run_id = safe_path_component(run_id, "ROBODOJO_RUN_ID")
            self.run_id = run_id
            safe_task_name = safe_path_component(self.task_name, "task_name")
            safe_policy_name = safe_path_component(self.policy_name, "policy_name")
            safe_config_name = safe_path_component(self.config_name, "config_name")
            safe_additional_info = safe_path_component(
                self.additional_info,
                "additional_info",
                allow_empty=True,
            )
            eval_seed_component = safe_path_component(
                f"{self.eval_seed}_{safe_additional_info}", "eval seed"
            )
            expected_parent = Path(
                "eval_result",
                f"{BENCHMARK}",
                safe_task_name,
                safe_policy_name,
                safe_config_name,
                eval_seed_component,
            )
            self.save_dir = os.path.join(
                "eval_result",
                f"{BENCHMARK}",
                safe_task_name,
                safe_policy_name,
                safe_config_name,
                eval_seed_component,
                run_id,
            )

            if resume_state is not None:
                resumed_save_dir = resume_state.get("save_dir")
                if resumed_save_dir:
                    self.save_dir = str(validate_resume_save_dir(resumed_save_dir, expected_parent))

            # Temp dir for in-progress streaming videos. Orphans left behind by
            # a hard kill (SIGKILL/crash) can't be cleaned via the in-memory
            # writer dict, so sweep them from disk on startup: any *.tmp.mp4
            # here belongs to an unfinished episode that will be re-run.
            self._stream_dir = os.path.join(self.save_dir, "_stream")
            self._sweep_stream_dir()

            self.obs_config = deepcopy(self.eval_cfg.get("observation", {}))
            self.obs_config["save_dir"] = self.save_dir
            self.description_cfg = self.eval_cfg.get("description", dict())
            self.obs_manager = ObsManager(
                obs_config=self.obs_config,
                num_envs=self.num_envs,
                dt=self.dt,
                task_name=self.task_name,
                description_cfg=self.description_cfg,
                seeds_per_env=self.env_seed_list,
            )

            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            # Per-env streaming video writers: {env_idx: {camera_key: writer}}.
            # Replaces the old full-episode frame cache; only vision frames are
            # streamed to disk as they arrive instead of buffered in RAM.
            self.video_writers: dict[int, dict[str, VideoStreamWriter]] = {}
            self.episode_nums = self.num_envs
            self.unstable_nums = 0
            self.unstable_envs: set[int] = set()
            self.success_nums = 0
            self.fail_nums = 0
            self.total_score = 0

            self.abandoned_seeds: set[int] = set()
            self.current_env_seed_map: dict[int, int] = {}
            self.eval_result = {
                "success_rate": 0.0,
                "eval_time": 0,
                "score": 0.0,
                "details": {},
                "rdv_abandoned_layouts": [],
            }

            self.scene_manager.layout_manager.replay = True
            self.seed_manager = SeedManager(config.eval_cfg)

            completed_layout_ids: list[int] = []
            abandoned_layout_ids: list[int] = []
            if resume_state is not None:
                self.success_nums = int(resume_state.get("success_nums", 0))
                self.fail_nums = int(resume_state.get("fail_nums", 0))
                self.total_score = float(resume_state.get("total_score", 0.0))

                resumed_details = resume_state.get("details") or {}

                normalised_details = {}
                for k, v in resumed_details.items():
                    try:
                        normalised_details[int(k)] = v
                    except (TypeError, ValueError):
                        normalised_details[k] = v
                self.eval_result["details"] = normalised_details
                self.abandoned_seeds = set(int(s) for s in resume_state.get("abandoned_layout_ids", []))
                completed_layout_ids = [
                    int(v["layout_id"]) for v in normalised_details.values() if isinstance(v, dict) and "layout_id" in v
                ]
                abandoned_layout_ids = list(self.abandoned_seeds)
                eval_time = self.success_nums + self.fail_nums
                if eval_time > 0:
                    self.eval_result["success_rate"] = self.success_nums / eval_time
                    self.eval_result["score"] = self.total_score / eval_time * 100
                self.eval_result["eval_time"] = eval_time
                print(
                    f"[EvalEnv][resume] save_dir={self.save_dir} "
                    f"success={self.success_nums} fail={self.fail_nums} "
                    f"completed={len(completed_layout_ids)} "
                    f"abandoned={len(abandoned_layout_ids)}"
                )
            self.seed_manager.init_eval(
                completed_layout_ids=completed_layout_ids,
                abandoned_layout_ids=abandoned_layout_ids,
            )

            self.deploy_cfg = config.deploy_cfg
            self.port = self.deploy_cfg.get("port", None)
            if self.port is None:
                raise ValueError("Port must be specified in deploy_cfg for the policy server!")
            self.host = self.deploy_cfg.get("host", "localhost")
            _patch_websockets_proxy_compat()

            policy_server_url = self.deploy_cfg.get("policy_server_url") or f"ws://{self.host}:{self.port}"
            evaluation_id = self.deploy_cfg.get("evaluation_id", self.run_id)
            trial_id = self.deploy_cfg.get("trial_id", f"{self.task_name}-{self.run_id}")
            action_case_id = self.deploy_cfg.get("action_case_id", f"{self.task_name}_case")
            self.model_client = WsModelClient(
                url=policy_server_url,
                evaluation_id=evaluation_id,
                trial_id=trial_id,
                action_case_id=action_case_id,
                repeat_index=self.deploy_cfg.get("repeat_index"),
                ws_ping_interval_s=self.deploy_cfg.get("ws_ping_interval_s", 20.0),
                ws_ping_timeout_s=self.deploy_cfg.get("ws_ping_timeout_s", 20.0),
            )
            self.robot_action_dim_info = get_robot_action_dim_info(env_cfg=self.eval_cfg)

        def close(self):
            self._abort_video_writers()
            super().close()

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        def reset(self, seed=None, options=None):
            if seed is None:
                seed = self.env_seeds
                if seed is None:
                    seed = self.seed_manager.get_seeds(max_count=self.num_envs)
                seed = [] if seed is None else list(seed)
            elif isinstance(seed, (int, np.integer)):
                seed = [int(seed)]
            else:
                try:
                    seed = list(seed)
                except TypeError as exc:
                    raise TypeError("seed must be None, an integer, or a sequence of integers") from exc
            if len(seed) < self.num_envs:
                seed = seed + [None] * (self.num_envs - len(seed))

            real_indices = [i for i, s in enumerate(seed) if s is not None]
            safe_seed = seed[real_indices[0]] if real_indices else 0
            # Fill None positions with safe_seed so scene_manager can still load
            self.env_seeds = [s if s is not None else safe_seed for s in seed]

            self.success = [True] * self.num_envs
            self.end_flag = [False] * self.num_envs
            self.take_action_cnt = [0] * self.num_envs
            # Discard any writers left open by a previous (e.g. crashed or
            # unstable) batch before starting a fresh one.
            self._abort_video_writers()
            self.episode_nums = len(real_indices)
            self.unstable_envs = set()

            self.current_env_seed_map = {}
            for idx in range(self.num_envs):
                self.scene_manager.layout_manager.set_saved_layout(
                    idx, self.seed_manager.get_seed_scene_info(self.env_seeds[idx])
                )
                if seed[idx] is None:
                    self.success[idx] = False
                    self.end_flag[idx] = True
                else:
                    self.current_env_seed_map[idx] = seed[idx]

            super().reset(seed=self.env_seeds, options=options)
            self.obs_manager.reset()  # Reset observation manager for the next episode
            self.setup_scene()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()

            self.model_client.call(func_name="reset")
            active_envs = [idx for idx in range(self.num_envs) if not self.end_flag[idx]]
            return self.get_obs_batch(env_idx_list=active_envs), {}

        def setup_scene(self):
            self.scene_manager.apply_saved_poses(env_idx_list=list(range(self.num_envs)))
            self._align_layout_success()
            success, unstable_envs = self.scene_manager.layout_manager.check_layout_stability(self)
            unstable_envs = [idx for idx in set(unstable_envs) if idx < self.num_envs]
            if not success and not unstable_envs:
                unstable_envs = [idx for idx in self.current_env_seed_map if not self.end_flag[idx]]
            self.unstable_nums += len(unstable_envs)
            self.unstable_envs.update(unstable_envs)
            for env_idx in unstable_envs:
                self.success[env_idx] = False
                self.end_flag[env_idx] = True
            for _ in range(10):
                self.render()
            for idx in range(200):
                self.sim_step()
                if idx % 5 == 0:
                    self.obs_manager.render_for_capture()
                    self.obs_manager.get_obs()
            if self.contact_monitor is not None:
                self.contact_monitor.clear_buffers()
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()

        def get_obs(self):
            return self.get_obs_batch(env_idx_list=[0])[0]

        def get_obs_batch(self, env_idx_list=None, last_frame=False):
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))
            if self.physx_monitor_enabled:
                self._check_endpose_finite(env_idx_list)
            self.obs_manager.render_for_capture()
            data = self.obs_manager.get_obs(env_idx_list=env_idx_list)
            data_list = []
            for env_idx in env_idx_list:
                if not self.end_flag[env_idx] or last_frame:
                    self._stream_vision(env_idx, data[env_idx])
                env_data = deepcopy(data[env_idx])
                env_data["env_idx"] = env_idx
                data_list.append(env_data)
            return data_list

        def eval_one_episode(self):
            policy_name = self.deploy_cfg["policy_name"]
            try:
                eval_module = __import__(
                    f"XPolicyLab.policy.{policy_name}.deploy",
                    fromlist=["eval_one_episode"],
                )
            except ImportError as e:
                print(
                    "[TestEnv]",
                    f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {e}",
                    "ERROR",
                )
                raise e

            if not hasattr(eval_module, "eval_one_episode"):
                print(
                    "[TestEnv]",
                    f"Module '.{policy_name}.deploy' does not have 'eval_one_episode' function",
                    "ERROR",
                )
                raise AttributeError("Missing eval_one_episode in policy module")

            eval_module.eval_one_episode(TASK_ENV=self, model_client=self.model_client)

        def eval_one_episode_batch(self):
            policy_name = self.deploy_cfg["policy_name"]
            try:
                eval_module = __import__(
                    f"XPolicyLab.policy.{policy_name}.deploy",
                    fromlist=["eval_one_episode_batch"],
                )
            except ImportError as e:
                print(
                    "[TestEnv]",
                    f"Failed to import policy module: XPolicyLab.policy.{policy_name}.deploy. Error: {e}",
                    "ERROR",
                )
                raise e

            if not hasattr(eval_module, "eval_one_episode_batch"):
                print(
                    "[TestEnv]",
                    f"Module '.{policy_name}.deploy' does not have 'eval_one_episode_batch' function",
                    "ERROR",
                )
                raise AttributeError("Missing eval_one_episode_batch in policy module")

            eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

        def get_action_type(self, action):
            action_type = []
            for robot in self.robot_manager.robot_list:
                if robot.type != "target":
                    continue
                key_name = self.robot_manager.process_name(robot.arm_name)
                if key_name in action.keys() and "joint" not in action_type:
                    action_type.append("joint")
                key_name = f"{robot.arm_name.split('_')[0]}_ee_pose"
                if key_name in action.keys() and "ee" not in action_type:
                    action_type.append("ee")
            if len(action_type) == 0:
                raise ValueError(f"Cannot infer action type from action dict keys: {action.keys()}")
            if len(action_type) > 1:
                raise ValueError(f"Multiple action types found in action dict keys: {action.keys()}")
            return action_type[0]

        def take_action(self, action):
            normalized = self.validate_action_dict(action)
            self.take_action_batch([normalized], env_idx_list=[0])

        def take_action_batch(self, actions_list, env_idx_list=None):
            if self.physx_monitor_enabled:
                self._check_physx_broken_envs()
            control_info_list = []
            action_env_idx_list = []
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))
            if len(actions_list) != len(env_idx_list):
                raise ValueError(
                    f"actions_list length {len(actions_list)} does not match env_idx_list length {len(env_idx_list)}"
                )
            validated_actions = []
            for idx, env_idx in enumerate(env_idx_list):
                normalized_action = self.validate_action_dict(actions_list[idx])
                action_type = self.get_action_type(normalized_action)
                validated_actions.append((env_idx, normalized_action, action_type))

            for env_idx, action, action_type in validated_actions:
                if self.take_action_cnt[env_idx] == self.step_lim or self.end_flag[env_idx]:
                    continue

                self.take_action_cnt[env_idx] += 1
                print(
                    f"env{env_idx} step: \033[92m{self.take_action_cnt[env_idx]} / {self.step_lim}\033[0m",
                    end="\r",
                )
                control_info = dict()
                if action_type == "joint":
                    for robot in self.robot_manager.robot_list:
                        if robot.type != "target":
                            continue
                        key_name = self.robot_manager.process_name(robot.arm_name)
                        control_info[key_name] = {
                            "position": action[key_name],
                        }

                        key_name = self.robot_manager.process_name(robot.gripper_name)
                        if robot.ee_type == "gripper":
                            val = deepcopy(action[key_name][0])
                            val = np.clip(val, 0, 1)
                            if robot.gripper_move["sign"] == 1:
                                val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                            else:
                                val = (1 - val) * (
                                    robot.gripper_scale[1] - robot.gripper_scale[0]
                                ) + robot.gripper_scale[0]
                            vals = [
                                val,
                                val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                            ]
                            control_info[key_name] = {"position": vals}
                        elif robot.ee_type == "hand":
                            control_info[key_name] = {
                                "position": action[key_name],
                            }
                        else:
                            pass
                elif action_type == "ee":
                    for robot in self.robot_manager.robot_list:
                        if robot.type != "target":
                            continue
                        name = robot.arm_name.split("_")[0]
                        key_name = f"{name}_ee_pose"
                        obs_name = self.robot_manager.process_name(robot.arm_name)
                        target_pose = action[key_name]
                        ik_result = self.robot_manager.solve_ik(
                            target_pose=target_pose,
                            env_idx=env_idx,
                            robot=robot,
                        )
                        if ik_result["status"] == "Success":
                            control_info[obs_name] = {
                                "position": ik_result["joint_value"],
                            }

                        key_name = self.robot_manager.process_name(robot.gripper_name)
                        if robot.ee_type == "gripper":
                            val = deepcopy(action[key_name][0])
                            val = np.clip(val, 0, 1)
                            if robot.gripper_move["sign"] == 1:
                                val = val * (robot.gripper_scale[1] - robot.gripper_scale[0]) + robot.gripper_scale[0]
                            else:
                                val = (1 - val) * (
                                    robot.gripper_scale[1] - robot.gripper_scale[0]
                                ) + robot.gripper_scale[0]
                            vals = [
                                val,
                                val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                            ]
                            control_info[key_name] = {"position": vals}
                        elif robot.ee_type == "hand":
                            control_info[key_name] = {
                                "position": action[key_name],
                            }
                        else:
                            pass
                control_seq = self.process_control_info(control_info, env_idx)
                control_info_list.append(control_seq)
                action_env_idx_list.append(env_idx)

            with self.reward_action_context(action_env_idx_list):
                if len(control_info_list) > 0:
                    self.robot_manager.control_manager.push(action_env_idx_list, control_info_list)
                    while not self.have_empty(action_env_idx_list):
                        self.step(env_idx_list=action_env_idx_list)
                        if self.physx_monitor_enabled:
                            self._check_physx_broken_envs()
                            self._check_endpose_finite(action_env_idx_list)

                if action_env_idx_list:
                    self.reward_manager.step(env_idx_list=action_env_idx_list)
            if getattr(self, "interact", False):
                if hasattr(self, "query_support_arm_traj"):
                    for env_idx in action_env_idx_list:
                        self.query_support_arm_traj(env_idx=env_idx)
                if hasattr(self, "check_support_arm_stable"):
                    for env_idx in action_env_idx_list:
                        self.check_support_arm_stable(env_idx=env_idx)
            self.is_episode_end()

        def mark_env_unstable(self, env_idx):
            """Flag an env as unstable so it is recorded as a failed sample."""
            self.unstable_envs.add(env_idx)

        def process_control_info(self, control_info, env_idx):
            """Expand one-step `control_info` into a per-step control sequence.

            This method returns a list with `interpolation_nums` control dicts
            (deep copies of the input), then overwrites arm/gripper positions
            frame by frame:

            - First 80% (floor) steps: linear interpolation from the current
                arm state or previously committed gripper command to the target.
            - Last 20% steps: hold the target state.

            Notes:
            - Arm joints are interpolated in joint space.
            - For grippers, interpolation is done on the primary scalar opening,
                then mapped to mimic joints.
            - Gripper values are clamped to `robot.gripper_scale` to avoid
                out-of-range commands.

            Args:
                    control_info: One-step target control dict for a single env.
                    env_idx: Environment index used to read robot state and the
                        previously committed gripper command.

            Returns:
                    List[dict]: A control sequence of length `interpolation_nums`.
            """
            interpolation_nums = int(self.obs_manager.collect_interval)
            if interpolation_nums <= 0:
                return [deepcopy(control_info)]

            control_info_list = [deepcopy(control_info) for _ in range(interpolation_nums)]
            for robot in self.robot_manager.robot_list:
                if robot.type != "target":
                    if hasattr(self, "support_arm_action") and len(self.support_arm_action[env_idx]) > 0:
                        for i in range(interpolation_nums):
                            if len(self.support_arm_action[env_idx]) == 0:
                                break
                            control_info_list[i].update(self.support_arm_action[env_idx][0])
                            self.support_arm_action[env_idx].pop(0)

                key_name = self.robot_manager.process_name(robot.arm_name)
                if key_name in control_info.keys():
                    position = control_info[key_name]["position"]
                    current_position = self.robot_manager.get_joint(robot, env_idx_list=[env_idx])[env_idx]
                    if current_position is not None:
                        interp_count = int(np.floor(interpolation_nums * 0.8))

                        current_arr = np.array(current_position)
                        target_arr = np.array(position)

                        for i in range(interp_count):
                            alpha = (i + 1) / (interp_count + 1)
                            interp_pos = (1 - alpha) * current_arr + alpha * target_arr
                            control_info_list[i][key_name]["position"] = interp_pos.tolist()

                        for i in range(interp_count, interpolation_nums):
                            control_info_list[i][key_name]["position"] = target_arr.tolist()

                key_name = self.robot_manager.process_name(robot.gripper_name)
                if key_name in control_info.keys() and robot.ee_type == "gripper":
                    position = control_info[key_name]["position"][0]
                    current_position = self.robot_manager.control_manager.get_last_commanded_gripper(
                        env_idx, key_name
                    )
                    if current_position is None:
                        measured_position = self.robot_manager.get_end_effector_real_val(
                            robot, env_idx_list=[env_idx]
                        )[env_idx]
                        current_position = measured_position[0] if measured_position is not None else None
                    if current_position is not None:
                        interp_count = int(np.floor(interpolation_nums * 0.8))

                        scale = robot.gripper_scale
                        for i in range(interp_count):
                            alpha = (i + 1) / (interp_count + 1)
                            interp_pos = (1 - alpha) * current_position + alpha * position
                            interp_pos = np.clip(interp_pos, scale[0], scale[1])
                            vals = [
                                interp_pos,
                                interp_pos * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                            ]
                            control_info_list[i][key_name]["position"] = vals

                        position = np.clip(position, scale[0], scale[1])
                        vals = [
                            position,
                            position * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2],
                        ]
                        for i in range(interp_count, interpolation_nums):
                            control_info_list[i][key_name]["position"] = vals

            return control_info_list

        @staticmethod
        def _has_bool_or_non_real_numeric(obj) -> bool:
            if isinstance(obj, (bool, np.bool_)):
                return True
            if isinstance(obj, (int, float, np.integer, np.floating)):
                return False
            if isinstance(obj, np.ndarray):
                if obj.dtype == np.bool_:
                    return True
                if obj.dtype == object:
                    return any(EvalEnv._has_bool_or_non_real_numeric(x) for x in obj.flat)
                return not (np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.floating))
            if isinstance(obj, (list, tuple)):
                return any(EvalEnv._has_bool_or_non_real_numeric(x) for x in obj)
            return True

        def validate_action_dict(self, action_dict: dict) -> dict:
            """Validate and normalize a policy action dict for this robot.

            Args:
                action_dict: action mapping returned by the policy. For a
                    single arm, use ``arm_joint_state`` or ``arm_ee_pose``.
                    Dual-arm targets use ``left_`` and ``right_`` prefixes.

            Returns:
                Normalized dict mapping each required key to a 1D float64
                ``np.ndarray``.

            Raises:
                ValueError: unexpected keys, forbidden prefixes, wrong
                    dimensions, non-finite values, or zero-norm quaternions.
                TypeError: ``action_dict`` is not a dict or a value is not a
                    real-numeric array-like (e.g. strings or booleans).
            """
            if not isinstance(action_dict, dict):
                raise TypeError(f"action_dict must be a dict, got {type(action_dict)}")

            arm_dims = deepcopy(self.robot_action_dim_info["arm_dim"])
            ee_dims = deepcopy(self.robot_action_dim_info["ee_dim"])

            if len(arm_dims) != len(ee_dims):
                raise ValueError(
                    f"robot_action_dim_info mismatch: len(arm_dim)={len(arm_dims)} != len(ee_dim)={len(ee_dims)}"
                )

            target_robots = [robot for robot in self.robot_manager.robot_list if robot.type == "target"]
            if len(target_robots) != len(arm_dims):
                raise ValueError(
                    f"robot/action mismatch: {len(target_robots)} target arms but {len(arm_dims)} configured arms"
                )

            expected = {}
            joint_keys = []
            ee_keys = []
            ee_pose_keys = set()
            for index, robot in enumerate(target_robots):
                arm_key = self.robot_manager.process_name(robot.arm_name)
                gripper_key = self.robot_manager.process_name(robot.gripper_name)
                pose_key = f"{robot.arm_name.split('_')[0]}_ee_pose"
                expected.update({arm_key: arm_dims[index], gripper_key: ee_dims[index], pose_key: 7})
                joint_keys.extend((arm_key, gripper_key))
                ee_keys.extend((pose_key, gripper_key))
                ee_pose_keys.add(pose_key)

            unexpected_keys = [k for k in action_dict if k not in expected]
            if unexpected_keys:
                raise ValueError(f"Unexpected state keys: {unexpected_keys}")

            has_joint = any(key in action_dict for key in joint_keys[::2])
            has_ee = any(key in action_dict for key in ee_keys[::2])
            if has_joint and has_ee:
                raise ValueError("Action cannot mix joint-state and end-effector pose keys")
            required_keys = joint_keys if has_joint else ee_keys if has_ee else []
            if not required_keys:
                raise ValueError("Action must contain a complete joint-state or end-effector-pose command")
            missing_keys = [key for key in required_keys if key not in action_dict]
            if missing_keys:
                raise ValueError(f"Action is missing required keys: {missing_keys}")
            extra_keys = [key for key in action_dict if key not in required_keys]
            if extra_keys:
                raise ValueError(f"Action mixes unsupported keys for this action type: {extra_keys}")

            normalized = {}
            for key, expected_dim in expected.items():
                if key not in required_keys:
                    continue
                value = action_dict[key]

                if not isinstance(value, (np.ndarray, list, tuple)):
                    raise TypeError(f"action_dict['{key}'] must be array-like, got {type(value)}")

                if self._has_bool_or_non_real_numeric(value):
                    raise TypeError(f"action_dict['{key}'] must contain real numeric values (no bool/str)")

                try:
                    arr = np.asarray(value, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"action_dict['{key}'] must contain numeric values") from exc

                if arr.ndim != 1:
                    raise ValueError(f"action_dict['{key}'] must be 1D, got shape {arr.shape}")

                if arr.shape[0] != expected_dim:
                    raise ValueError(
                        f"action_dict['{key}'] dim mismatch: expected {expected_dim}, got shape {arr.shape}"
                    )
                if not np.isfinite(arr).all():
                    raise ValueError(f"action_dict['{key}'] must contain only finite values")

                if key in ee_pose_keys:
                    quat_norm = float(np.linalg.norm(arr[3:7]))
                    if quat_norm <= 1e-6:
                        raise ValueError(
                            f"action_dict['{key}'] quaternion must have non-zero norm, got {arr[3:7].tolist()}"
                        )

                normalized[key] = arr

            return normalized

        def step(self, env_idx_list, decimation=1):
            meta_control_list = self.robot_manager.control_manager.pop(env_idx_list)
            for _ in range(decimation):
                super().step(meta_control_list=meta_control_list)
                self.sim_step(render=False)

        def _align_layout_success(self):
            for env_idx in range(self.num_envs):
                if self.end_flag[env_idx]:
                    continue
                if not self.scene_manager.layout_manager.layout_valid[env_idx]:
                    self.success[env_idx] = False
                    self.end_flag[env_idx] = True
                    self.unstable_envs.add(env_idx)

        def resume_manifest_path(self) -> str:
            """Stable-but-run-id-tagged path for the resume manifest.

            Lives one directory above the timestamped save_dir so that
            independent eval invocations (each with their own ROBODOJO_RUN_ID)
            never overwrite each other's manifest while still being easy to
            locate by humans.
            """
            safe_task_name = safe_path_component(self.task_name, "task_name")
            safe_policy_name = safe_path_component(self.policy_name, "policy_name")
            safe_config_name = safe_path_component(self.config_name, "config_name")
            safe_additional_info = safe_path_component(
                self.additional_info, "additional_info", allow_empty=True
            )
            eval_seed_component = safe_path_component(
                f"{self.eval_seed}_{safe_additional_info}", "eval seed"
            )
            return os.path.join(
                "eval_result",
                f"{BENCHMARK}",
                safe_task_name,
                safe_policy_name,
                safe_config_name,
                eval_seed_component,
                f"_resume_{self.run_id}.json",
            )

        def persist_resume_manifest(self, restart_count: int = 0) -> str:
            """Atomically persist enough state to resume after process death.

            Writes ``_resume_<run_id>.json`` next to the timestamped save_dir.
            Always called at the end of run_eval() (best-effort) and again
            from main.py's PhysXFatalError handler (authoritative). Atomic
            via tmp + rename so a partial write cannot corrupt resume.
            """
            manifest_path = self.resume_manifest_path()
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            self.eval_result["rdv_abandoned_layouts"] = sorted(int(s) for s in self.abandoned_seeds)
            completed_layout_ids = sorted(
                {
                    int(v["layout_id"])
                    for v in self.eval_result.get("details", {}).values()
                    if isinstance(v, dict) and "layout_id" in v
                }
            )
            payload = {
                "run_id": self.run_id,
                "save_dir": self.save_dir,
                "task_name": self.task_name,
                "policy_name": self.policy_name,
                "config_name": self.config_name,
                "eval_seed": self.eval_seed,
                "additional_info": self.additional_info,
                "success_nums": int(self.success_nums),
                "fail_nums": int(self.fail_nums),
                "unstable_nums": int(self.unstable_nums),
                "total_score": float(self.total_score),
                "completed_layout_ids": completed_layout_ids,
                "abandoned_layout_ids": sorted(int(s) for s in self.abandoned_seeds),
                "details": self.eval_result.get("details", {}),
                "restart_count": int(restart_count),
            }
            tmp_path = manifest_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, default=str)
            os.replace(tmp_path, manifest_path)
            return manifest_path

        def _check_physx_broken_envs(self):
            if not self.physx_monitor_enabled:
                return
            # Fatal kernel-failure short-circuit: the GPU solver has died and
            # nothing inside this process can recover. Surface immediately
            # so main.py can persist progress and re-exec.
            monitor = self._physx_get_monitor()
            if monitor.is_fatal():
                raise self._PhysXFatalError(monitor.get_fatal_message())
            bad = monitor.get_broken_envs()
            active_slots = getattr(self, "current_env_seed_map", None)
            if active_slots:
                new_bad = {i for i in bad if i in active_slots and not self.end_flag[i]}
            else:
                new_bad = {i for i in bad if i < self.num_envs and not self.end_flag[i]}
            if new_bad:
                raise self._PhysXBrokenError(new_bad)

        def _check_endpose_finite(self, env_idx_list):
            """NaN backstop run right before obs_manager.get_obs.

            Carb's PhysX warnings are captured by PhysXWarningMonitor via fd
            interception. Kit/Carb rebinds its logger fd across
            env.close()/recreate cycles, so the monitor occasionally misses
            warnings. When a miss happens, the next obs_manager.get_obs
            feeds a NaN-laden 3x3 into mat2quat, which throws LinAlgError.
            main.py's generic except then sees an empty broken-env set,
            falls through to seed_manager.eval_step(), and silently consumes
            the whole batch of seeds.

            This method recomputes the exact 3x3 that get_delta_endpose is
            about to hand to mat2quat and verifies np.isfinite on it. Any env
            that fails the check is treated as PhysX-broken and we raise
            PhysXBrokenError so main.py records that layout as a failed sample
            and retries only the healthy envs in the batch.
            """
            if not self.physx_monitor_enabled:
                return
            bad = set()
            for robot in self.robot_manager.robot_list:
                poses = self.robot_manager.get_real_endpose(robot, env_idx_list=env_idx_list, is_relative=True)
                for env_idx in env_idx_list:
                    if env_idx >= self.num_envs or self.end_flag[env_idx]:
                        continue
                    ee_pose = poses.get(env_idx)
                    if ee_pose is None:
                        continue
                    if not np.isfinite(ee_pose).all():
                        bad.add(env_idx)
                        continue
                    try:
                        rot_3x3 = t3d.quaternions.quat2mat(ee_pose[-4:]) @ robot.delta_matrix
                    except Exception:
                        bad.add(env_idx)
                        continue
                    if not np.isfinite(rot_3x3).all():
                        bad.add(env_idx)
            if bad:
                self._physx_get_monitor().add_broken_envs(bad)
                raise self._PhysXBrokenError(bad)

        def get_seeds_for_envs(self, env_idxs) -> set:
            return {self.current_env_seed_map[i] for i in env_idxs if i in self.current_env_seed_map}

        def run_eval(self):
            self.run_reward()
            if hasattr(self, "get_score"):
                self.get_score()
            batch_envs = list(self.current_env_seed_map)
            exist_envs = self.get_running_env_idx_list()
            if exist_envs:
                if getattr(self, "interact", False):
                    if hasattr(self, "query_support_arm_traj"):
                        for env_idx in exist_envs:
                            self.query_support_arm_traj(env_idx=env_idx)
                if self.eval_batch:
                    self.eval_one_episode_batch()
                else:
                    self.eval_one_episode()
            # A policy is allowed to return without calling take_action_batch.
            # Force the task's terminal checks here so the initial success
            # sentinel cannot turn an immediate quit into a 100-point episode.
            active_after_policy = [env_idx for env_idx in exist_envs if not self.end_flag[env_idx]]
            if active_after_policy:
                final_rewards = self.reward_manager.get_reward(
                    final_check=True, env_idx_list=active_after_policy
                )
                for env_idx in active_after_policy:
                    self.end_flag[env_idx] = True
                    self.success[env_idx] = final_rewards[env_idx] > 1 - 1e-3
            success = 0
            process_scores = (
                self.reward_manager.get_score(
                    reward_lst=[1.0 if self.success[env_idx] else 0.0 for env_idx in range(self.num_envs)]
                )
                if hasattr(self, "get_score")
                else None
            )
            recorded = 0
            for env_idx in batch_envs:
                index = self.success_nums + self.fail_nums + recorded
                episode_score = 0.0
                tag = "fail"
                is_unstable = env_idx in self.unstable_envs
                if is_unstable:
                    self.success[env_idx] = False
                if self.success[env_idx]:
                    self.total_score += 1.0
                    episode_score = 1.0
                    success += 1
                    tag = "success"
                elif process_scores is not None and not is_unstable:
                    episode_score = process_scores[env_idx] / 100.0
                    self.total_score += episode_score

                # seed_list was filtered by completed/abandoned ids on resume,
                # so seed_list.index(seed) no longer yields the original
                # layout id. Since init_eval populates seed_list as
                # range(N_layouts), seed == layout_id by construction; use
                # env_seeds[env_idx] directly.
                self.eval_result["details"][index] = {
                    "layout_id": int(self.env_seeds[env_idx]),
                    "success": bool(self.success[env_idx]),
                    "score": episode_score,
                }
                if env_idx not in self.unstable_envs:
                    video_path = os.path.join(self.save_dir, f"episode_{index:07d}.mp4")
                    self.save_video(env_idx, video_path, tag)
                recorded += 1

            # Drop streams for envs not saved this batch (e.g. unstable ones).
            self._abort_video_writers()

            fail = recorded - success
            self.success_nums += success
            self.fail_nums += fail
            eval_time = self.success_nums + self.fail_nums
            if eval_time > 0:
                self.eval_result["success_rate"] = self.success_nums / eval_time
                self.eval_result["score"] = self.total_score / eval_time * 100
            self.eval_result["eval_time"] = eval_time
            save_json(self.eval_result, os.path.join(self.save_dir, "_result.json"))
            # Refresh the resume manifest at the end of every batch so that a
            # downstream SIGABRT (which beats the in-process PhysXFatalError
            # handler) still recovers everything up to the previous batch.
            try:
                self.persist_resume_manifest()
            except Exception as e:
                print(f"[EvalEnv] persist_resume_manifest after run_eval failed: {e}")

        def record_batch_failure(self, error: Exception, env_idxs=None) -> None:
            """Count a non-PhysX batch exception as failed eval samples."""
            selected = None if env_idxs is None else set(int(idx) for idx in env_idxs)
            failed_envs = []
            for env_idx, seed in self.current_env_seed_map.items():
                if selected is not None and env_idx not in selected:
                    continue
                failed_envs.append((env_idx, seed))

            reason = f"{type(error).__name__}: {error}"[:500]
            start_index = self.success_nums + self.fail_nums
            for offset, (env_idx, seed) in enumerate(failed_envs):
                self.success[env_idx] = False
                self.end_flag[env_idx] = True
                self.eval_result["details"][start_index + offset] = {
                    "layout_id": int(seed),
                    "success": False,
                    "score": 0.0,
                    "error": reason,
                }

            self.fail_nums += len(failed_envs)
            self.episode_nums = max(0, self.episode_nums - len(failed_envs))
            self._abort_video_writers()
            eval_time = self.success_nums + self.fail_nums
            self.eval_result["eval_time"] = eval_time
            self.eval_result["success_rate"] = self.success_nums / eval_time if eval_time else 0.0
            self.eval_result["score"] = self.total_score / eval_time * 100 if eval_time else 0.0
            save_json(self.eval_result, os.path.join(self.save_dir, "_result.json"))
            try:
                self.persist_resume_manifest()
            except Exception as persist_error:
                print(f"[EvalEnv] persist_resume_manifest after batch failure failed: {persist_error}")

        def is_episode_end(self):
            active_envs = [env_idx for env_idx in range(self.num_envs) if not self.end_flag[env_idx]]
            if not active_envs:
                return True
            pre_end_flag = deepcopy(self.end_flag)
            final_envs = [
                env_idx
                for env_idx in active_envs
                if self.take_action_cnt[env_idx] >= self.step_lim or not self.success[env_idx]
            ]
            reward_list = self.reward_manager.get_reward(final_check=False, env_idx_list=active_envs)
            if final_envs:
                final_reward_list = self.reward_manager.get_reward(
                    final_check=True, env_idx_list=final_envs
                )
                for env_idx in final_envs:
                    reward_list[env_idx] = final_reward_list[env_idx]
            for env_idx in range(self.num_envs):
                if self.end_flag[env_idx]:
                    continue
                if reward_list[env_idx] > 1 - 1e-3:
                    self.end_flag[env_idx] = True
                    self.success[env_idx] = True
                    continue
                if self.take_action_cnt[env_idx] >= self.step_lim or not self.success[env_idx]:
                    self.end_flag[env_idx] = True
                    self.success[env_idx] = False

            end_flag_changed_list = [
                env_idx for env_idx in range(self.num_envs) if self.end_flag[env_idx] != pre_end_flag[env_idx]
            ]
            if len(end_flag_changed_list) > 0:
                self.get_obs_batch(env_idx_list=end_flag_changed_list, last_frame=True)
            return all(self.end_flag)

        def get_running_env_idx_list(self):
            return [idx for idx in range(self.num_envs) if not self.end_flag[idx]]

        def _stream_vision(self, env_idx, frame):
            """Append this env's per-camera RGB frames to its ffmpeg streams.

            Only the vision ("color") data is recorded; writers are created
            lazily on the first frame (when the resolution is known) and write
            to temporary files until the episode outcome decides the name.
            """
            vision = frame.get("vision") if isinstance(frame, dict) else None
            if not vision:
                return
            writers = self.video_writers.setdefault(env_idx, {})
            fps = self.obs_manager.collect_freq
            for cam_key, cam_data in vision.items():
                color = cam_data.get("color") if isinstance(cam_data, dict) else None
                if color is None:
                    continue
                color = np.ascontiguousarray(color)
                if color.ndim != 3 or color.shape[2] not in (3, 4):
                    continue
                if cam_key not in writers:
                    height, width, channels = color.shape
                    tmp_path = os.path.join(self._stream_dir, f"env{env_idx}_{cam_key}.tmp.mp4")
                    # Keep only a small, bounded number of snapshots per
                    # camera so recording cannot grow RAM usage during long
                    # episodes. The writer blocks the producer when this
                    # bound is reached instead of dropping frames.
                    writers[cam_key] = VideoStreamWriter(
                        tmp_path,
                        height,
                        width,
                        channels,
                        fps=fps,
                        queue_size=2,
                    )
                writers[cam_key].append(color)

        def _sweep_stream_dir(self):
            """Remove orphan temp videos left by a previous hard kill/crash."""
            stream_dir = getattr(self, "_stream_dir", None)
            if not stream_dir or not os.path.isdir(stream_dir):
                return
            for name in os.listdir(stream_dir):
                if name.endswith(".tmp.mp4"):
                    try:
                        os.remove(os.path.join(stream_dir, name))
                    except Exception:
                        pass

        def _abort_video_writers(self, env_idx_list=None):
            """Close and delete partial videos for the given (or all) envs."""
            if env_idx_list is None:
                env_idx_list = list(self.video_writers.keys())
            for env_idx in list(env_idx_list):
                writers = self.video_writers.pop(env_idx, {})
                for writer in writers.values():
                    try:
                        writer.abort()
                    except Exception:
                        pass

        def save_video(self, env_idx, video_path, tag):
            writers = self.video_writers.pop(env_idx, {})
            # Signal EOF for every camera before waiting for any one encoder.
            # This lets the independent ffmpeg workers drain concurrently.
            for writer in writers.values():
                try:
                    writer.request_close()
                except Exception:
                    pass
            for cam_key, writer in writers.items():
                tmp_path = writer.out_path
                final_path = video_path.replace(".mp4", f"_{cam_key}_{tag}.mp4")
                try:
                    writer.close(announce=False)
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    os.replace(tmp_path, final_path)
                except Exception as e:
                    print(f"[EvalEnv] Failed to finalize video for env {env_idx} cam {cam_key}: {e}")
                    writer.abort()
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    continue
                print(
                    format_video_saved_message(
                        final_path,
                        writer.n_frames,
                        writer.width,
                        writer.height,
                        writer.fps,
                    )
                )

        def have_empty(self, env_idx_list=None):
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))
            return len(self.get_control_empty(env_idx_list=env_idx_list)) != 0

        def get_control_empty(self, env_idx_list=None):
            if env_idx_list is None:
                env_idx_list = list(range(self.num_envs))
            return self.robot_manager.control_manager.get_empty(env_idx_list=env_idx_list)

    return EvalEnv(config, app, resume_state=resume_state, **kwargs)
