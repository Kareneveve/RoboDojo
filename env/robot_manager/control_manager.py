from copy import deepcopy
import queue

import numpy as np

from env.environment.isaac.physx_device import ensure_host_joint_targets


class MetaControl:  # Single control signal for one env step
    BASE_CONTROL_KEY = "base_pose_state"

    def __init__(self, control_info_dict=None):
        self.gripper_velocity_constant = 1.0
        self.control_info_dict = control_info_dict if control_info_dict is not None else dict()
        self._action_cache = {}

    def get_action(self, robot_manager, env_idx):
        if any(key in self.control_info_dict for key in ("qpos", "qvel", self.BASE_CONTROL_KEY)):
            cache_key = (id(robot_manager), int(env_idx))
            if cache_key in self._action_cache:
                return self._action_cache[cache_key]
            mixed_keys = [
                key for key in robot_manager.get_robot_obs_name() if self.control_info_dict.get(key) is not None
            ]
            if mixed_keys:
                raise ValueError(f"Named joint targets cannot be mixed with component controls: {mixed_keys}.")
            action = {}
            for key in ("qpos", "qvel"):
                values = self.control_info_dict.get(key, {}) or {}
                action[key] = {name: _joint_scalar(value, name) for name, value in values.items()}
            base_control = self.control_info_dict.get(self.BASE_CONTROL_KEY)
            if base_control is not None:
                base_control = deepcopy(base_control)
                velocity = np.asarray(base_control.get("velocity"), dtype=np.float64)
                if base_control.get("control_mode") != "direct_velocity":
                    raise ValueError("Navigation control requires direct_velocity base control.")
                if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
                    raise ValueError("Direct root velocity must be finite shape (6,).")
                base_control["velocity"] = velocity.copy()
                action[self.BASE_CONTROL_KEY] = base_control
            self._action_cache[cache_key] = action
            return action

        def get_dict(control_info_key):
            position = self.control_info_dict[control_info_key]["position"]
            velocity = (
                self.control_info_dict[control_info_key]["velocity"]
                if "velocity" in self.control_info_dict[control_info_key]
                else 0
            )
            return {"position": position, "velocity": velocity}

        def process_gripper_val(robot_manager, robot, position, gripper_eps=0.2, env_idx=None):
            real_gripper_val = robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx]
            real_gripper_val = real_gripper_val[0]
            scale = robot.gripper_scale
            percentage = np.abs(position - real_gripper_val) / (scale[1] - scale[0])
            if percentage < gripper_eps:
                val = position
            else:
                if position > real_gripper_val:
                    val = real_gripper_val + (scale[1] - scale[0]) * gripper_eps
                else:
                    val = real_gripper_val - (scale[1] - scale[0]) * gripper_eps
            return [val, val * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]

        res = dict()
        obs_list = robot_manager.get_robot_obs_name()
        for key in obs_list:
            if key in self.control_info_dict.keys():
                if key.endswith("ee_joint_state"):
                    end_effector_name = robot_manager.restore_name(deepcopy(key))
                    robot = robot_manager.get_robot_by_gripper_name(end_effector_name)
                    if robot.ee_type == "gripper":
                        position = get_dict(key)["position"]
                        gripper_val = process_gripper_val(
                            robot_manager, robot, position[0], gripper_eps=0.2, env_idx=env_idx
                        )
                        res[key] = {"position": gripper_val, "velocity": get_dict(key)["velocity"]}
                    else:
                        res[key] = get_dict(key)
                else:
                    res[key] = get_dict(key)
            else:
                res[key] = None
        return res


class ControlSeq:  # Control sequence for one environment
    def __init__(self):
        self.control_seq = queue.Queue()

    def push(self, control_list):  # list
        for control_info in control_list:
            self.control_seq.put(
                control_info if isinstance(control_info, MetaControl) else MetaControl(control_info)
            )

    def pop(self):
        assert not self.is_empty(), "The Control Seq is Empty"
        return self.control_seq.get()

    def is_empty(self):
        return self.control_seq.empty()


class ControlManager:  # Control sequences for all environments
    def __init__(self, num_envs, robot_manager):
        self.num_envs = num_envs
        self.robot_manager = robot_manager
        self.control_queue = [ControlSeq() for _ in range(self.num_envs)]  # create queue
        self.prev_control = [dict() for _ in range(self.num_envs)]
        self.joint_pos_target = [dict() for _ in range(self.num_envs)]
        self.joint_vel_target = [dict() for _ in range(self.num_envs)]

    def update_prev_control(self, env_idx, new_meta_ctrl: MetaControl):
        meta_ctrl_dict = new_meta_ctrl.get_action(self.robot_manager, env_idx)
        if "qpos" in meta_ctrl_dict:
            for robot in self.robot_manager.robot_list:
                for name, joints in (
                    (robot.arm_name, robot.arm_joints_name),
                    (robot.gripper_name, robot.gripper_joints_name),
                ):
                    names = self.robot_manager._canonical_action_joint_names(joints, robot)
                    self._merge_joint_group(
                        self.prev_control[env_idx],
                        self.robot_manager.process_name(name),
                        names,
                        meta_ctrl_dict,
                    )
            return
        obs_list = self.robot_manager.get_robot_obs_name()
        for key in obs_list:
            if meta_ctrl_dict[key] is not None:
                self.prev_control[env_idx][key] = meta_ctrl_dict[key]
        if self.joint_pos_target[env_idx] or self.joint_vel_target[env_idx]:
            self._update_named_targets_from_components(env_idx, meta_ctrl_dict)

    def update_current_missing_ctrl_info(self, env_idx, new_meta_ctrl: MetaControl):
        cached = getattr(new_meta_ctrl, "_pop_merge_cache", None)
        if cached is not None and cached[0] == env_idx:
            return cached[1]
        meta_ctrl_dict = new_meta_ctrl.get_action(self.robot_manager, env_idx)
        if "qpos" in meta_ctrl_dict:
            self._initialize_joint_targets(env_idx)
            positions = self.joint_pos_target[env_idx]
            velocities = self.joint_vel_target[env_idx]
            for key, cache in (("qpos", positions), ("qvel", velocities)):
                for name, value in meta_ctrl_dict[key].items():
                    cache[self._resolve_cached_joint_name(cache, name)] = value
            merged = {"qpos": deepcopy(positions), "qvel": deepcopy(velocities)}
            if MetaControl.BASE_CONTROL_KEY in meta_ctrl_dict:
                merged[MetaControl.BASE_CONTROL_KEY] = deepcopy(meta_ctrl_dict[MetaControl.BASE_CONTROL_KEY])
            merged_control = MetaControl(merged)
            new_meta_ctrl._pop_merge_cache = (env_idx, merged_control)
            return merged_control
        obs_list = self.robot_manager.get_robot_obs_name()
        for key in obs_list:
            if meta_ctrl_dict[key] is None:
                meta_ctrl_dict[key] = self.prev_control[env_idx][key]
        return MetaControl(meta_ctrl_dict)

    def _update_named_targets_from_components(self, env_idx, action):
        """Keep an existing named snapshot current when a component frame runs."""
        for robot in self.robot_manager.robot_list:
            for name, joints in (
                (robot.arm_name, robot.arm_joints_name),
                (robot.gripper_name, robot.gripper_joints_name),
            ):
                state = action.get(self.robot_manager.process_name(name))
                if state is None:
                    continue
                names = self.robot_manager._canonical_action_joint_names(joints, robot)
                for field, cache in (
                    ("position", self.joint_pos_target[env_idx]),
                    ("velocity", self.joint_vel_target[env_idx]),
                ):
                    values = np.broadcast_to(_target_array(state[field]), (len(names),))
                    for joint_name, value in zip(names, values, strict=True):
                        cache[self._resolve_cached_joint_name(cache, joint_name)] = float(value)

    @staticmethod
    def _merge_joint_group(previous, key, names, action):
        if not any(name in action["qpos"] for name in names):
            return
        prior_state = previous.get(key)
        if prior_state is None:
            if not all(name in action["qpos"] for name in names):
                return
            prior_state = {"position": [action["qpos"][name] for name in names]}
            previous[key] = prior_state
        else:
            prior_state["position"] = [
                action["qpos"].get(name, value)
                for name, value in zip(names, prior_state["position"], strict=True)
            ]
        if any(name in action["qvel"] for name in names):
            velocities = np.broadcast_to(_target_array(prior_state.get("velocity", 0.0)), (len(names),))
            prior_state["velocity"] = [
                action["qvel"].get(name, value) for name, value in zip(names, velocities, strict=True)
            ]

    def _initialize_joint_targets(self, env_idx):
        """Snapshot actuator targets once, preserving uncommanded body joints."""
        if self.joint_pos_target[env_idx]:
            return
        articulations = []
        for articulation in self.robot_manager.robot_key:
            if all(articulation is not other for other in articulations):
                articulations.append(articulation)
        controlled = {id(articulation): self._controlled_joint_ids(articulation) for articulation in articulations}
        owners = {}
        for articulation in articulations:
            for joint_id in controlled[id(articulation)]:
                name = articulation.joint_names[joint_id]
                owners[name] = owners.get(name, 0) + 1
        velocity_only_names = set(getattr(self.robot_manager, "velocity_only_joint_names", ()))
        for index, articulation in enumerate(articulations):
            host_targets = ensure_host_joint_targets(articulation)
            positions = _target_array(
                (host_targets["joint_pos_target"] if host_targets is not None else articulation.data.joint_pos_target)[env_idx]
            )
            velocities = _target_array(
                (host_targets["joint_vel_target"] if host_targets is not None else articulation.data.joint_vel_target)[env_idx]
            )
            for joint_id in controlled[id(articulation)]:
                name = articulation.joint_names[joint_id]
                action_name = f"robot{index}/{name}" if owners[name] > 1 else name
                if name not in velocity_only_names:
                    self.joint_pos_target[env_idx][action_name] = float(positions[joint_id])
                self.joint_vel_target[env_idx][action_name] = float(velocities[joint_id])

    @staticmethod
    def _controlled_joint_ids(articulation):
        joint_count = len(articulation.joint_names)
        controlled = set()
        for actuator in getattr(articulation, "actuators", {}).values():
            indices = actuator.joint_indices
            if isinstance(indices, slice):
                controlled.update(range(joint_count)[indices])
            elif hasattr(indices, "tolist"):
                values = indices.tolist()
                controlled.update([int(values)] if isinstance(values, int) else (int(value) for value in values))
            else:
                controlled.update(int(value) for value in indices)
        return sorted(controlled) if controlled else list(range(joint_count))

    @staticmethod
    def _resolve_cached_joint_name(cache, name):
        if name in cache:
            return name
        matches = [key for key in cache if key.endswith(f"/{name}")]
        return matches[0] if len(matches) == 1 else name

    def get_empty(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        result = []
        for i in range(self.num_envs):
            if i not in env_idx_list:
                while not self.control_queue[i].is_empty():
                    self.control_queue[i].pop()
            else:
                if self.control_queue[i].is_empty():
                    result.append(i)
        return result

    def push(self, env_idx_list, control_queue_list):
        assert len(env_idx_list) == len(control_queue_list), (
            f"[ERROR] Mismatching Input List Len: {len(env_idx_list)} != {len(control_queue_list)}"
        )
        for i_th in range(len(env_idx_list)):
            idx = env_idx_list[i_th]
            self.control_queue[idx].push(control_queue_list[i_th])

    def pop(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(range(self.num_envs))
        empty_queue_list = self.get_empty(env_idx_list)
        assert len(empty_queue_list) == 0, f"[ERROR] Empty Control Queue: {empty_queue_list}"
        res = []
        for env_idx in range(self.num_envs):
            if env_idx not in env_idx_list:
                meta_ctrl = []
            else:
                meta_ctrl = self.control_queue[env_idx].pop()
                meta_ctrl = self.update_current_missing_ctrl_info(env_idx, meta_ctrl)
                self.update_prev_control(env_idx, meta_ctrl)
            res.append(meta_ctrl)
        return res

    def reset(self):  # clean all queue
        self.control_queue = [ControlSeq() for _ in range(self.num_envs)]  # create queue
        self.prev_control = [dict() for _ in range(self.num_envs)]
        self.joint_pos_target = [dict() for _ in range(self.num_envs)]
        self.joint_vel_target = [dict() for _ in range(self.num_envs)]


def _joint_scalar(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.size != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"Joint target {name!r} must be a finite scalar.")
    return float(value.reshape(-1)[0])


def _target_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)
