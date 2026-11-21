from contextlib import contextmanager
from typing import Any, List

from env.camera_manager.camera_manager import CameraManager
from env.environment.base_env import BaseEnv
from env.robot_manager.control_manager import ControlSeq
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.scene_manager import SceneManager


class TaskEnv(BaseEnv):
    def __init__(self, config, app, **kwargs):
        # Optional contact/motion sensors; tasks opt in via enable_*_monitor().
        self.contact_monitor = None
        self.placement_monitor = None
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cpu")
        self.dt = self.sim_config.dt

        # RobotManager must be created before BaseEnv.__init__ because it
        # provides the InteractiveSceneCfg used to launch the simulation.
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            config=self.robot_config,
            dt=self.dt,
            device=self.device,
        )
        self._interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing
        )
        super().__init__(self.sim_config, app=app, interactiveSceneCfg=self._interactiveSceneCfg, **kwargs)
        self.env_seed_list = self.env_seeds

        self.scene_manager = SceneManager(
            self.num_envs,
            self.device,
            self.sim_config.scene.env_spacing,
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )

        self.robot_manager._load_camera_cfg(config.camera)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = CameraManager(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager

        self.capture_manager = TiledCaptureManager(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    def enable_contact_monitor(self, labels, velocity_labels=(), contact_pairs=None):
        """Track per-step contacts between ``labels`` and pre-step body states.

        Call from a task ``__init__``.  The monitor is started after the scene
        is created, re-resolved on every reset, samples every ``sim_step`` and
        is stopped on ``close``.  Reward predicates read it via
        ``env.contact_monitor``.  ``contact_pairs`` defaults to every pair of
        ``labels``.  Contact-report events additionally need the objects to
        enable contact reporting, e.g. with the ``contact_report`` override.
        """
        from env.scene_manager.contact_monitor import ContactMonitor

        self.contact_monitor = ContactMonitor(
            self.num_envs, labels=labels, velocity_labels=velocity_labels, contact_pairs=contact_pairs
        )
        return self.contact_monitor

    def enable_placement_monitor(self, item_supports, min_contact_impulse=1e-5):
        """Track fingertip grasps and item-support contacts over each action window.

        Call from a task ``__init__``.  The monitor is started after the scene
        is created, refreshed on every reset, scoped to each action through
        ``reward_action_context``, and stopped on ``close``.  Monitored items
        must enable contact reporting via ``SceneManager.set_physics_overrides``.
        """
        from env.scene_manager.placement_monitor import PlacementContactMonitor

        self.placement_monitor = PlacementContactMonitor(
            self.num_envs, item_supports=item_supports, min_contact_impulse=min_contact_impulse
        )
        return self.placement_monitor

    @contextmanager
    def reward_action_context(self, env_idx_list):
        """Lifecycle hook wrapped around one policy action and its reward step."""
        if self.placement_monitor is None:
            yield
            return
        self.placement_monitor.begin_action(env_idx_list, ended_flags=getattr(self, "end_flag", None))
        try:
            yield
            reward_manager = getattr(self, "reward_manager", None)
            func_parser = getattr(reward_manager, "func_parser", None)
            self.placement_monitor.snapshot_positions(env_idx_list, func_parser)
        finally:
            self.placement_monitor.end_action()

    def sim_step(self, render: bool = True):
        if self.contact_monitor is not None:
            self.contact_monitor.capture_pre_step()
        return super().sim_step(render=render)

    def _setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        super()._setup_scene(sim)

    def _post_setup_scene(self, sim):
        self.scene_manager.initialize(sim)
        if self.contact_monitor is not None:
            self.contact_monitor.start(sim, self.scene_manager, dt=self.dt)
        if self.placement_monitor is not None:
            self.placement_monitor.start(sim, self.scene_manager, self.robot_manager)
        super()._post_setup_scene(sim)
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        self.camera_manager.post_init()

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        if self.contact_monitor is not None:
            self.contact_monitor.invalidate()
        if self.placement_monitor is not None:
            self.placement_monitor.invalidate()
        self.scene_manager.reload_scene()
        self.robot_manager.reset()
        for _ in range(300):
            self.sim_step(render=False)
        self.camera_manager.reset()
        self.capture_manager.reset()
        if self.contact_monitor is not None:
            self.contact_monitor.reset()
        if self.placement_monitor is not None:
            self.placement_monitor.reset()

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        if self.placement_monitor is not None:
            self.placement_monitor.stop()
        if self.contact_monitor is not None:
            self.contact_monitor.stop()
        self.capture_manager.destroy()
        self.camera_manager.destroy()
        self.robot_manager.close()
        self.scene_manager.close()
        super().close()
