import os
import re

from omegaconf import OmegaConf
import yaml

from env.benchmark import BenchmarkPaths
from env.global_configs import *
from utils.load_file import *


def get_embodiment_config(robot_name, key=None):
    if key is not None:
        config_path = os.path.join(ROBOTS_PATH, f"{robot_name}/{key}_robot_config.yml")
    else:
        config_path = os.path.join(ROBOTS_PATH, f"{robot_name}/robot_config.yml")
    with open(config_path, encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_embodiment_config_by_robot_type(robot_type, robot_name, key=None):
    robot_args = dict()
    robot_args["robot_config"] = get_embodiment_config(robot_name, key=key)
    robot_args["robot_file"] = os.path.join(ROBOTS_PATH, robot_name)
    robot_args["robot_name"] = robot_name
    robot_args["robot_type"] = robot_type
    return robot_args


def get_robot_action_dim_info(env_cfg):
    robot_config_name = env_cfg["config"]["robot"]
    robot_action_dim_info = load_json(os.path.join(ENV_CONFIG_PATH, "robot", "_robot_info.json"))[robot_config_name]
    return robot_action_dim_info


def process_randomization(env_cfg):
    randomization_cfg = env_cfg["eval_cfg"].get("domain_randomization", {})
    if "scene" in env_cfg and "Table" in env_cfg["scene"]:
        if randomization_cfg.get("random_table", False):
            env_cfg["scene"]["Table"]["random"] = True
        else:
            env_cfg["scene"]["Table"]["random"] = False

    if "scene" in env_cfg and "Ground" in env_cfg["scene"]:
        materials = env_cfg["scene"]["Ground"].get("materials")
        if materials is not None:
            materials["random"] = bool(randomization_cfg.get("random_ground", False))

    if "scene" in env_cfg and "Background" in env_cfg["scene"]:
        if randomization_cfg.get("random_background", False):
            env_cfg["scene"]["Background"]["random"] = True
        else:
            env_cfg["scene"]["Background"]["random"] = False

    return env_cfg


def resolve_random_task_num_envs(task_name, num_envs, sim_cfg):
    """Cap parallel env count for *_random tasks using sim.scene.clutter_env_limit."""
    if not str(task_name).endswith("_random"):
        return int(num_envs)

    scene_cfg = sim_cfg.get("scene", {}) if sim_cfg is not None else {}
    limit = scene_cfg.get("clutter_env_limit", 5)
    return min(int(num_envs), int(limit))

def _task_setting(task_info, common_info, key, default):
    return task_info.get(key, common_info.get(key, default))


def _enable_teleop_physx_stabilization(sim_cfg):
    sim_cfg.setdefault("physx", {})
    sim_cfg["physx"]["enable_stabilization"] = True


def process_config(env_cfg, task_name):
    benchmark_paths = BenchmarkPaths.from_config(env_cfg.get("eval_cfg", {}))
    task_index_path = benchmark_paths.config_dir / "_task.yml"
    info = load_yaml(task_index_path)
    task_info = info["tasks"].get(task_name, {})
    common_info = info.get("common", {})

    # Apply declared task overrides; keep existing simulation defaults otherwise.
    if "device" in task_info or "device" in common_info:
        device = str(_task_setting(task_info, common_info, "device", "cpu")).strip().lower()
        if re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
            raise ValueError(f"device must be cpu, cuda, or cuda:<index>, got {device!r} for task {task_name!r}.")
        OmegaConf.update(env_cfg, "sim.device", device, force_add=True)

    if "overwrite_gpu_setting" in task_info or "overwrite_gpu_setting" in common_info:
        overwrite_gpu_setting = int(_task_setting(task_info, common_info, "overwrite_gpu_setting", 0))
        if overwrite_gpu_setting not in (-1, 0, 1):
            raise ValueError(
                "overwrite_gpu_setting must be -1, 0, or 1, "
                f"got {overwrite_gpu_setting} for task {task_name!r}."
            )
        OmegaConf.update(env_cfg, "sim.overwrite_gpu_setting", overwrite_gpu_setting, force_add=True)

    if _task_setting(task_info, common_info, "data_source", "datagen") == "teleop":
        _enable_teleop_physx_stabilization(env_cfg["sim"])

    default_render_interval = 10
    render_interval = _task_setting(task_info, common_info, "render_interval", default_render_interval)
    if render_interval != default_render_interval:
        env_cfg["sim"]["render_interval"] = render_interval

    default_scene = "default"
    scene_config = _task_setting(task_info, common_info, "scene_config", default_scene)
    if scene_config != default_scene:
        env_cfg.scene = load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", scene_config + ".yml"))

    default_camera = "camera_config"
    camera_config = _task_setting(task_info, common_info, "camera_config", default_camera)
    if camera_config != default_camera:
        env_cfg.camera = load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", camera_config + ".yml"))

    default_robot = "dual_x5"
    robot_config = _task_setting(task_info, common_info, "robot_config", default_robot)
    if robot_config != default_robot:
        env_cfg.robot = load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", robot_config + ".yml"))

    if not _task_setting(task_info, common_info, "robot_self_collision", True):
        for cfg in env_cfg.robot.robots:
            cfg["enabled_self_collisions"] = False

    eval_num = task_info.get("eval_nums", common_info.get("eval_nums", 50))
    return env_cfg, eval_num
