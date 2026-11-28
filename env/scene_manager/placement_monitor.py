"""Action-windowed grasp and support-contact tracking for placement tasks.

``PlacementContactMonitor`` records, over each policy action window:

* whether a monitored item is pinched between both gripper fingers of any
  target robot on any physics substep (so held objects are not mistaken for
  released placements);
* the env-local contact points between each item and its expected support
  surface (including the scene table when ``support == "table"``).

``TaskEnv.enable_placement_monitor`` owns the lifecycle and wraps each action
via ``TaskEnv.reward_action_context``.  Reward predicates query it through
``env.placement_monitor`` (see ``Func_Parser.is_released_and_supported``).
Monitored items must author ``PhysxContactReportAPI``, e.g. via
``SceneManager.set_physics_overrides({label: {"contact_report": True}})``.

Carb/Isaac/PhysX modules are imported lazily so this module stays importable
in unit tests without a running simulator.
"""

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from env.scene_manager.contact_monitor import (
    CONTACT_PROCESSING_SETTING as CONTACT_PROCESSING_SETTING,
    acquire_contact_processing,
    ensure_contact_processing_enabled,
    get_env_origin,
    release_contact_processing,
    to_numpy,
)

DEFAULT_MIN_CONTACT_IMPULSE = 1e-5  # N·s per contact point


class PlacementContactMonitor:
    """Track fingertip grasps and item-support contacts per action window."""

    def __init__(
        self,
        num_envs: int,
        item_supports: Mapping[str, str],
        min_contact_impulse: float = DEFAULT_MIN_CONTACT_IMPULSE,
    ):
        self.num_envs = int(num_envs)
        self.item_supports = {str(item): str(support) for item, support in item_supports.items()}
        self.min_contact_impulse = float(min_contact_impulse)

        self.scene_manager = None
        self.robot_manager = None
        self._sim = None
        self._previous_post_physics_step_callback = None
        self._reporting_acquired = False
        self._bound = False
        self._clear_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, sim, scene_manager, robot_manager):
        """Enable PhysX contact processing and attach the substep callback."""
        self.stop()
        self.scene_manager = scene_manager
        self.robot_manager = robot_manager
        self._enable_reporting()
        self._sim = sim
        self._previous_post_physics_step_callback = getattr(sim, "post_physics_step_callback", None)
        sim.post_physics_step_callback = self._on_post_physics_step
        self._clear_state()

    def invalidate(self):
        """Clear bound actors and buffers before a scene reload."""
        self._bound = False
        self._clear_state()

    def reset(self):
        """Re-enable contact processing and clear buffers after a scene reload."""
        self.invalidate()
        self._enable_reporting()
        if self.scene_manager is not None:
            self.bind_scene()

    def stop(self):
        """Restore the previous post-physics callback and contact setting."""
        self.invalidate()
        if (
            self._sim is not None
            and getattr(self._sim, "post_physics_step_callback", None) == self._on_post_physics_step
        ):
            self._sim.post_physics_step_callback = self._previous_post_physics_step_callback
        self._sim = None
        self._previous_post_physics_step_callback = None
        self._restore_reporting()

    # ------------------------------------------------------------------
    # Action window & queries
    # ------------------------------------------------------------------
    def bind_scene(self):
        """Resolve item, support, table, and gripper-finger prim paths."""
        self._actors.clear()
        self._origins.clear()
        if self.scene_manager is None:
            self._bound = False
            return

        scene = self.scene_manager
        layout = scene.layout_manager
        object_labels = tuple(
            dict.fromkeys([*self.item_supports, *(s for s in self.item_supports.values() if s != "table")])
        )
        uses_table = "table" in self.item_supports.values()

        for env_idx, root in enumerate(scene.env_roots):
            paths = {}
            for label in object_labels:
                name = layout.get_instance_name(env_idx=env_idx, label=label)
                obj = layout.get_scene_object(env_idx, name) if name is not None else None
                prim_path = (
                    getattr(obj, "prim_path", None) or getattr(obj, "usd_prim_path", None) if obj is not None else None
                )
                if prim_path:
                    paths[str(prim_path)] = (env_idx, label)
            if len(paths) != len(object_labels):
                continue

            origin = get_env_origin(getattr(scene, "env_origins", None), env_idx)
            if origin is None:
                continue
            self._origins[env_idx] = origin

            if uses_table:
                paths[f"{root}/Table"] = (env_idx, "table")
                table = scene._tables[env_idx] if env_idx < len(scene._tables) else None
                table_path = getattr(table, "_prim_path", None) or getattr(table, "prim_path", None)
                if table_path:
                    paths[str(table_path)] = (env_idx, "table")

            if self.robot_manager is not None:
                for robot in self.robot_manager.robot_list:
                    if getattr(robot, "type", None) != "target":
                        continue
                    robot_prim = robot.SceneCfg.prim_path.rsplit("/", 1)[-1]
                    finger_links = getattr(robot, "gripper_finger_links", ("link7", "link8"))
                    for finger in finger_links:
                        paths[f"{root}/{robot_prim}/{finger}"] = (env_idx, (robot_prim, finger))

            self._actors.update(paths)

        self._bound = bool(self._origins)

    def begin_action(self, env_idx_list, ended_flags=None):
        """Start collecting substep contacts for the given active environments."""
        if not self._bound:
            self.bind_scene()
        self._active = {
            idx for idx in env_idx_list if idx in self._origins and (ended_flags is None or not ended_flags[idx])
        }
        for idx in env_idx_list:
            self._held[idx].clear()
            self._support_points[idx].clear()

    def end_action(self):
        """Stop collecting contacts until the next action window."""
        self._active.clear()

    def snapshot_positions(self, env_idx_list, func_parser):
        """Record monitored item poses in ``func_parser.pre_state`` for the next step."""
        if self.scene_manager is None or func_parser is None:
            return
        layout = self.scene_manager.layout_manager
        for env_idx in env_idx_list:
            previous = func_parser.pre_state[env_idx]
            for label in self.item_supports:
                name = layout.get_instance_name(env_idx=env_idx, label=label)
                if name is None:
                    continue
                pos, quat = layout.get_instance_pose(env_idx=env_idx, inst_name=name)
                if pos is not None and quat is not None:
                    previous.setdefault(name, {})["pose"] = np.concatenate((to_numpy(pos), to_numpy(quat)))

    def is_held(self, env_idx: int, label: str) -> bool:
        """Return whether ``label`` was pinched by a robot gripper during the action."""
        return str(label) in self._held[int(env_idx)]

    def get_support_points(self, env_idx: int, label: str, support: str) -> list[np.ndarray]:
        """Return env-local contact points between ``label`` and ``support``."""
        if self.item_supports.get(str(label)) != str(support):
            return []
        return list(self._support_points[int(env_idx)].get(str(label), ()))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clear_state(self):
        self._active: set[int] = set()
        self._actors: dict[str, tuple] = {}
        self._origins: dict[int, np.ndarray] = {}
        self._held: list[set[str]] = [set() for _ in range(self.num_envs)]
        self._support_points: list[dict[str, list[np.ndarray]]] = [{} for _ in range(self.num_envs)]

    def _enable_reporting(self):
        if self._reporting_acquired:
            ensure_contact_processing_enabled()
        else:
            self._reporting_acquired = acquire_contact_processing()

    def _restore_reporting(self):
        if not self._reporting_acquired:
            return
        self._reporting_acquired = False
        release_contact_processing()

    def _find_actor(self, path: str):
        while path:
            if path in self._actors:
                return self._actors[path]
            path = path.rpartition("/")[0]
        return None

    def _contact_points(self, data, header):
        start = header.contact_data_offset
        for contact in data[start : start + header.num_contact_data]:
            impulse = np.linalg.norm(np.asarray(contact.impulse, dtype=float))
            point = np.asarray(contact.position, dtype=float)
            if np.isfinite(impulse) and impulse >= self.min_contact_impulse and np.isfinite(point).all():
                yield point

    def record_substep(self):
        """Poll PhysX contacts for the substep that just completed."""
        if not self._active:
            return
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        headers, data = get_physx_simulation_interface().get_contact_report()
        # Keep fingertip pairs local to this substep.
        pad_contacts = defaultdict(set)
        for header in headers:
            if not header.num_contact_data:
                continue
            first = self._find_actor(str(PhysicsSchemaTools.intToSdfPath(header.actor0)))
            second = self._find_actor(str(PhysicsSchemaTools.intToSdfPath(header.actor1)))
            if first is None or second is None:
                continue
            env_idx, label = first
            other_env, other = second
            if env_idx != other_env or env_idx not in self._active:
                continue
            if label not in self.item_supports:
                label, other = other, label
            is_finger = isinstance(other, tuple)
            if label not in self.item_supports or (not is_finger and other != self.item_supports[label]):
                continue

            points = self._contact_points(data, header)
            if is_finger:
                robot, finger = other
                if next(points, None) is not None:
                    pad_contacts[env_idx, label, robot].add(finger)
            else:
                origin = self._origins[env_idx]
                self._support_points[env_idx].setdefault(label, []).extend(point - origin for point in points)

        for (env_idx, label, _), fingers in pad_contacts.items():
            if len(fingers) >= 2:
                self._held[env_idx].add(label)

    def _on_post_physics_step(self):
        previous = self._previous_post_physics_step_callback
        if previous is not None:
            previous()
        self.record_substep()
