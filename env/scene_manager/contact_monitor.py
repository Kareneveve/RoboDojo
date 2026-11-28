"""Per-environment, per-physics-step contact and motion tracking for scene objects.

``ContactMonitor`` is a task-agnostic sensor.  Before every ``sim_step`` it:

* samples the contacts of the step that just completed for each monitored
  label pair, from a filtered ``RigidContactView`` (GPU) and from PhysX
  contact-report events (CPU), in env-local coordinates;
* keeps the contacts of the step before that, so a predicate can tell a new
  contact (onset) from a contact that was already there (static support,
  pressing, dragging);
* records the body state (pose, COM linear velocity, angular velocity) of
  selected labels *before* the step, so the approach velocity of any body
  point (e.g. a mallet head) at the moment of contact can be computed.

Each step with a contact becomes a time-aligned *sample*; reward predicates
read and consume samples through ``env.contact_monitor``.  ``TaskEnv`` owns the
lifecycle (``TaskEnv.enable_contact_monitor``).  Contact-report events require
the objects to author ``PhysxContactReportAPI`` (``contact_report`` physics
override, see ``SceneManager.set_physics_overrides``).

Isaac/PhysX modules are imported lazily so this module stays importable in
unit tests without a running simulator.
"""

from collections.abc import Iterable
from itertools import combinations
import threading

import numpy as np

DEFAULT_MAX_SAMPLES = 128
DEFAULT_MAX_CONTACT_COUNT = 256
CONTACT_PROCESSING_SETTING = "/physics/disableContactProcessing"
_EMPTY_CONTACTS = (np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float))
_CONTACT_PROCESSING_LOCK = threading.Lock()
_CONTACT_PROCESSING_REFCOUNT = 0
_SAVED_CONTACT_PROCESSING_SETTING = None


def _get_carb_settings():
    """Return the Carb settings interface when running inside Isaac Sim."""
    try:
        import carb

        return carb.settings.get_settings()
    except Exception:
        return None


def acquire_contact_processing() -> bool:
    """Enable PhysX contact processing with process-wide reference counting."""
    global _CONTACT_PROCESSING_REFCOUNT, _SAVED_CONTACT_PROCESSING_SETTING
    settings = _get_carb_settings()
    if settings is None:
        return False
    with _CONTACT_PROCESSING_LOCK:
        if _CONTACT_PROCESSING_REFCOUNT == 0:
            previous = settings.get(CONTACT_PROCESSING_SETTING)
            _SAVED_CONTACT_PROCESSING_SETTING = True if previous is None else bool(previous)
        _CONTACT_PROCESSING_REFCOUNT += 1
        settings.set_bool(CONTACT_PROCESSING_SETTING, False)
    return True


def ensure_contact_processing_enabled() -> None:
    """Ensure contact processing stays enabled without changing the refcount."""
    settings = _get_carb_settings()
    if settings is not None:
        settings.set_bool(CONTACT_PROCESSING_SETTING, False)


def release_contact_processing() -> None:
    """Release a contact-processing hold and restore the original setting when idle."""
    global _CONTACT_PROCESSING_REFCOUNT, _SAVED_CONTACT_PROCESSING_SETTING
    with _CONTACT_PROCESSING_LOCK:
        if _CONTACT_PROCESSING_REFCOUNT <= 0:
            _CONTACT_PROCESSING_REFCOUNT = 0
            return
        _CONTACT_PROCESSING_REFCOUNT -= 1
        if _CONTACT_PROCESSING_REFCOUNT > 0:
            return
        saved = _SAVED_CONTACT_PROCESSING_SETTING
        _SAVED_CONTACT_PROCESSING_SETTING = None
    if saved is not None:
        settings = _get_carb_settings()
        if settings is not None:
            settings.set_bool(CONTACT_PROCESSING_SETTING, saved)


def to_numpy(value, dtype=float):
    """Convert a torch tensor / array-like into a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def get_env_origin(env_origins, env_idx):
    """Return the finite 3D origin of ``env_idx`` or ``None`` if unavailable."""
    if env_origins is None:
        return None
    try:
        origin = to_numpy(env_origins[int(env_idx)]).reshape(-1)[:3]
    except (IndexError, TypeError, ValueError):
        return None
    if origin.size != 3 or not np.all(np.isfinite(origin)):
        return None
    return origin


def quat_wxyz_to_matrix(quat):
    """Rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float).reshape(-1)[:4]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def create_contact_view(prim_path_A, prim_path_B, name, max_contact_count=DEFAULT_MAX_CONTACT_COUNT):
    """Create and initialize a ``RigidContactView`` of A filtered against B."""
    settings = _get_carb_settings()
    if settings is not None:
        settings.set_bool(CONTACT_PROCESSING_SETTING, False)
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.simulation_manager import SimulationManager

    view = RigidContactView(
        prim_paths_expr=[str(prim_path_A)],
        filter_paths_expr=[str(prim_path_B)],
        name=str(name).replace("/", "_"),
        prepare_contact_sensors=True,
        max_contact_count=int(max_contact_count),
    )
    view.initialize(physics_sim_view=SimulationManager.get_physics_sim_view())
    return view


def read_contact_points(contact_view, dt, env_origin=None):
    """Read the current-step contacts of a filtered ``RigidContactView``.

    Returns ``(positions, normal_forces)`` with env-local ``(N, 3)`` positions,
    or ``None`` when the view data is unavailable.  Only the slice of the
    shared contact buffer that belongs to the first sensor/filter pair is
    used; ignoring the pair indices can reuse stale contacts of another pair.
    """
    contact_data = contact_view.get_contact_force_data(dt=dt)
    if contact_data is None or len(contact_data) < 2:
        return None
    normal_forces = to_numpy(contact_data[0]).reshape(-1)
    positions = to_numpy(contact_data[1]).reshape(-1, 3)
    count = min(len(normal_forces), len(positions))
    begin, end = 0, count
    if len(contact_data) >= 6:
        pair_counts = to_numpy(contact_data[4]).astype(int, copy=False).reshape(-1)
        pair_starts = to_numpy(contact_data[5]).astype(int, copy=False).reshape(-1)
        if pair_counts.size == 0 or pair_starts.size == 0:
            return _EMPTY_CONTACTS
        pair_count, pair_start = int(pair_counts[0]), int(pair_starts[0])
        if pair_count <= 0 or pair_start < 0 or pair_start >= count:
            return _EMPTY_CONTACTS
        begin, end = pair_start, min(pair_start + pair_count, count)
    normal_forces = normal_forces[begin:end]
    positions = positions[begin:end]
    # PhysX reports world coordinates; functional points are env-local.
    if env_origin is not None:
        positions = positions - env_origin
    return positions, normal_forces


def read_contact_separations(contact_view, dt):
    """Read PhysX contact separations for the first sensor/filter pair.

    Isaac Sim exposes contact distances as the fourth item returned by
    ``RigidContactView.get_contact_force_data``.  A negative value means that
    the reported contact shapes overlap at the sampled solver step.  Older
    Isaac Sim versions may omit this item, in which case ``None`` is
    returned.  The pair-count slice mirrors :func:`read_contact_points` so a
    stale contact from another sensor pair is never attributed to this pair.
    """
    try:
        contact_data = contact_view.get_contact_force_data(dt=dt)
    except Exception:
        return None
    if contact_data is None or len(contact_data) < 4:
        return None
    separations = to_numpy(contact_data[3]).reshape(-1)
    if len(contact_data) < 6:
        return separations
    pair_counts = to_numpy(contact_data[4]).astype(int, copy=False).reshape(-1)
    pair_starts = to_numpy(contact_data[5]).astype(int, copy=False).reshape(-1)
    if pair_counts.size == 0 or pair_starts.size == 0:
        return np.zeros(0, dtype=float)
    pair_count, pair_start = int(pair_counts[0]), int(pair_starts[0])
    if pair_count <= 0 or pair_start < 0 or pair_start >= separations.size:
        return np.zeros(0, dtype=float)
    return separations[pair_start : min(pair_start + pair_count, separations.size)]


def point_velocity(state, point_local):
    """Velocity of a body-frame point from a recorded body state.

    ``state`` holds ``lin`` (COM linear velocity), ``ang`` (angular velocity,
    rad/s, world frame), ``quat`` (wxyz body orientation) and ``com_local``
    (COM in the body frame); ``v = v_com + w x R (p_local - com_local)``.
    """
    rotation = quat_wxyz_to_matrix(state["quat"])
    lever = rotation @ (np.asarray(point_local, dtype=float).reshape(3) - np.asarray(state["com_local"], dtype=float))
    return np.asarray(state["lin"], dtype=float) + np.cross(np.asarray(state["ang"], dtype=float), lever)


class ContactMonitor:
    """Time-aligned contact samples and pre-step body states per environment."""

    def __init__(
        self,
        num_envs: int,
        labels: Iterable[str],
        velocity_labels: Iterable[str] = (),
        contact_pairs: Iterable[tuple[str, str]] | None = None,
        max_samples: int = DEFAULT_MAX_SAMPLES,
    ):
        self.num_envs = int(num_envs)
        self.labels = tuple(dict.fromkeys(str(label) for label in labels))
        self.velocity_labels = tuple(dict.fromkeys(str(label) for label in velocity_labels))
        pairs = combinations(self.labels, 2) if contact_pairs is None else contact_pairs
        self.contact_pairs = tuple((str(a), str(b)) for a, b in pairs)
        self.max_samples = int(max_samples)

        self.scene_manager = None
        self.dt = 1.0 / 60.0
        self._env_ready: list[bool] = [False] * self.num_envs
        self._sim = None
        self._subscription = None
        self._previous_post_physics_step_callback = None
        self._reporting_acquired = False
        self._roots: dict[str, tuple[int, str]] = {}
        self._objects: dict[str, list] = {}
        self._com_local: dict[str, list] = {}
        self._views: list[dict] = [{} for _ in range(self.num_envs)]
        self._env_origins: list[np.ndarray | None] = [None] * self.num_envs
        self._clear_buffers()

    @property
    def ready(self) -> bool:
        """Return whether at least one environment has resolved its monitored objects."""
        return any(self._env_ready)

    def is_env_ready(self, env_idx: int) -> bool:
        """Return whether ``env_idx`` has resolved its monitored objects."""
        idx = int(env_idx)
        return 0 <= idx < self.num_envs and self._env_ready[idx]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, sim, scene_manager, dt: float | None = None, *, resolve: bool = True) -> bool:
        """Enable PhysX contact processing, subscribe to reports, and optionally resolve objects.

        Pass ``resolve=False`` when calling before the physics scene is active
        (e.g. during ``_post_setup_scene`` prior to ``sim.reset()``); objects
        and contact views will then be resolved on the subsequent ``reset()``.
        """
        self.stop()
        self.scene_manager = scene_manager
        if dt is not None and np.isfinite(float(dt)) and float(dt) > 0.0:
            self.dt = float(dt)
        self._enable_reporting()
        self._subscribe_contact_reports()
        self._sim = sim
        self._previous_post_physics_step_callback = getattr(sim, "post_physics_step_callback", None)
        sim.post_physics_step_callback = self._on_post_physics_step
        if resolve:
            self.reset()
        return self.ready

    def invalidate(self):
        """Drop resolved objects and views before the scene is reloaded.

        Objects may be destroyed and recreated during a reload; until
        ``reset()`` resolves the new ones, nothing is tracked.
        """
        self._env_ready = [False] * self.num_envs
        self._roots.clear()
        self._objects = {}
        self._com_local = {}
        self._views = [{} for _ in range(self.num_envs)]
        self._clear_buffers()

    def clear_buffers(self):
        """Clear accumulated contact samples/history without rebuilding views.

        Called after episode settling steps so settling contacts do not pollute
        step-0 monitoring state.
        """
        self._clear_buffers()

    def reset(self):
        """Re-enable contact processing, resolve objects/views, and clear buffers.

        Prim paths can change between episodes, so they are resolved again.
        """
        self.invalidate()
        self._enable_reporting()
        self._subscribe_contact_reports()
        if self.scene_manager is None:
            return
        self._register_objects()
        self._create_views()

    def stop(self):
        """Unsubscribe and restore the previous callback and contact setting."""
        self.invalidate()
        self._subscription = None
        if (
            self._sim is not None
            and getattr(self._sim, "post_physics_step_callback", None) == self._on_post_physics_step
        ):
            self._sim.post_physics_step_callback = self._previous_post_physics_step_callback
        self._sim = None
        self._previous_post_physics_step_callback = None
        self._restore_reporting()

    # ------------------------------------------------------------------
    # Per-step capture
    # ------------------------------------------------------------------
    def capture_pre_step(self):
        """Sample the step that just completed, then record pre-step body states."""
        self._sample_completed_step()
        for env_idx in range(self.num_envs):
            self._state_before_step[env_idx] = {
                label: self._read_body_state(env_idx, label) for label in self.velocity_labels
            }
        self._step += 1

    def get_body_state(self, env_idx: int, label: str):
        """Return the body state recorded before the latest step, or ``None``."""
        return self._state_before_step[int(env_idx)].get(str(label))

    def get_pre_step_velocity(self, env_idx: int, label: str):
        """Return the COM linear velocity recorded before the latest step, or ``None``."""
        state = self.get_body_state(env_idx, label)
        return None if state is None else np.asarray(state["lin"], dtype=float).copy()

    def get_contact_samples(self, env_idx: int, label_A: str, label_B: str) -> list[dict]:
        """Return the unconsumed contact samples of a pair, oldest first.

        The latest completed step is sampled first.  Each sample has ``step``,
        ``positions``/``forces`` (contacts of that step), ``prev_positions``/
        ``prev_forces`` (contacts of the step before) and ``states`` (body
        states recorded before that step, keyed by label).
        """
        self._sample_completed_step()
        key = frozenset((str(label_A), str(label_B)))
        return [sample for sample in self._samples[int(env_idx)] if sample["pair"] == key]

    def get_contact_separations(self, env_idx: int, label_A: str, label_B: str):
        """Return current PhysX separations for a monitored pair when available."""
        key = frozenset((str(label_A), str(label_B)))
        view = self._views[int(env_idx)].get(key)
        if view is None:
            return None
        return read_contact_separations(view, self.dt)

    def consume_samples(self, env_idx: int, label_A: str, label_B: str):
        """Discard all buffered samples of a pair."""
        key = frozenset((str(label_A), str(label_B)))
        env_idx = int(env_idx)
        self._samples[env_idx] = [sample for sample in self._samples[env_idx] if sample["pair"] != key]

    def has_contacted(self, env_idx: int, label_A: str, label_B: str) -> bool:
        """Return whether ``label_A`` and ``label_B`` have contacted since the last reset."""
        if not self.is_env_ready(env_idx):
            return False
        self._sample_completed_step()
        return frozenset((str(label_A), str(label_B))) in self._contacted_pairs[int(env_idx)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _enable_reporting(self):
        if self._reporting_acquired:
            ensure_contact_processing_enabled()
        else:
            self._reporting_acquired = acquire_contact_processing()

    def _subscribe_contact_reports(self):
        if self._subscription is not None:
            return
        try:
            from omni.physx import get_physx_simulation_interface

            self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report
            )
        except Exception as exc:
            print(f"[ContactMonitor] Failed to subscribe to contact reports: {exc}")
            self._subscription = None

    def _restore_reporting(self):
        if not self._reporting_acquired:
            return
        self._reporting_acquired = False
        release_contact_processing()

    def _clear_buffers(self):
        self._step = 0
        self._sampled_step = 0
        self._samples = [[] for _ in range(self.num_envs)]
        self._contacted_pairs = [set() for _ in range(self.num_envs)]
        self._pending_events = [[] for _ in range(self.num_envs)]
        self._step_events = [{} for _ in range(self.num_envs)]
        self._last_contacts = [{} for _ in range(self.num_envs)]
        self._state_before_step = [{} for _ in range(self.num_envs)]

    def _get_scene_object(self, env_idx, label):
        layout_manager = self.scene_manager.layout_manager
        inst_name = layout_manager.get_instance_name(label=label, env_idx=env_idx)
        if inst_name is None:
            return None
        return layout_manager.get_scene_object(inst_name=inst_name, env_idx=env_idx)

    def _register_objects(self) -> bool:
        """Resolve monitored objects, their root paths and COM offsets per env."""
        env_origins = getattr(self.scene_manager, "env_origins", None)
        self._env_origins = [get_env_origin(env_origins, env_idx) for env_idx in range(self.num_envs)]
        all_labels = tuple(dict.fromkeys(self.labels + self.velocity_labels))
        self._objects = {label: [None] * self.num_envs for label in all_labels}
        self._com_local = {label: [np.zeros(3)] * self.num_envs for label in self.velocity_labels}
        self._env_ready = [False] * self.num_envs
        for env_idx in range(self.num_envs):
            env_found = True
            for label in all_labels:
                try:
                    obj = self._get_scene_object(env_idx, label)
                except Exception:
                    obj = None
                prim_path = getattr(obj, "usd_prim_path", None) if obj is not None else None
                if not self._is_valid_prim_path(prim_path):
                    env_found = False
                    continue
                self._objects[label][env_idx] = obj
                if label in self.labels:
                    self._roots[str(prim_path)] = (env_idx, label)
                if label in self._com_local:
                    self._com_local[label][env_idx] = self._read_com_local(obj)
            self._env_ready[env_idx] = env_found
        return any(self._env_ready)

    def _create_views(self):
        for env_idx in range(self.num_envs):
            for label_A, label_B in self.contact_pairs:
                obj_A = self._objects.get(label_A, [None] * self.num_envs)[env_idx]
                obj_B = self._objects.get(label_B, [None] * self.num_envs)[env_idx]
                if obj_A is None or obj_B is None:
                    continue
                try:
                    self._views[env_idx][frozenset((label_A, label_B))] = create_contact_view(
                        obj_A.usd_prim_path,
                        obj_B.usd_prim_path,
                        name=f"contact_monitor_{env_idx}_{label_A}_{label_B}",
                    )
                except Exception as exc:
                    print(f"[ContactMonitor] Could not create contact view {label_A}/{label_B} in env {env_idx}: {exc}")

    @staticmethod
    def _is_valid_prim_path(prim_path) -> bool:
        if not prim_path:
            return False
        try:
            from isaacsim.core.utils.prims import get_prim_at_path

            prim = get_prim_at_path(str(prim_path))
            return prim is not None and prim.IsValid()
        except Exception:
            return False

    @staticmethod
    def _read_com_local(obj):
        try:
            com_positions, _ = obj._rigid_prim_view.get_coms()
            com = to_numpy(com_positions).reshape(-1, 3)[0]
            return com if np.all(np.isfinite(com)) else np.zeros(3)
        except Exception:
            return np.zeros(3)

    def _read_body_state(self, env_idx, label):
        obj = self._objects.get(label, [None] * self.num_envs)[env_idx]
        if obj is None:
            return None
        try:
            pos, quat = obj.get_local_pose()
            state = {
                "pos": to_numpy(pos).reshape(-1)[:3],
                "quat": to_numpy(quat).reshape(-1)[:4],
                "lin": to_numpy(obj.get_linear_velocity()).reshape(-1)[:3],
                "ang": to_numpy(obj.get_angular_velocity()).reshape(-1)[:3],
                "com_local": self._com_local[label][env_idx],
            }
        except Exception:
            return None
        if not all(np.all(np.isfinite(value)) for value in state.values()):
            return None
        return state

    def _read_step_contacts(self, env_idx, key):
        """Contacts of the latest completed step for one pair (view + events)."""
        positions, forces = [], []
        view = self._views[env_idx].get(key)
        if view is not None:
            try:
                points = read_contact_points(view, self.dt, self._env_origins[env_idx])
            except Exception:
                points = None
            if points is not None:
                positions.append(points[0])
                forces.append(points[1])
        events = self._step_events[env_idx].pop(key, [])
        if events:
            positions.append(np.array([event[0] for event in events], dtype=float).reshape(-1, 3))
            forces.append(np.array([event[1] for event in events], dtype=float))
        if not positions:
            return _EMPTY_CONTACTS
        positions = np.concatenate(positions).reshape(-1, 3)
        forces = np.concatenate(forces).reshape(-1)
        in_contact = np.isfinite(forces) & (np.abs(forces) > 0.0)
        return positions[in_contact], forces[in_contact]

    def _sample_completed_step(self):
        """Record the contacts of the latest completed step once."""
        if self._step == 0 or self._sampled_step == self._step:
            return
        self._sampled_step = self._step
        for env_idx in range(self.num_envs):
            for label_A, label_B in self.contact_pairs:
                key = frozenset((label_A, label_B))
                positions, forces = self._read_step_contacts(env_idx, key)
                prev_positions, prev_forces = self._last_contacts[env_idx].get(key, _EMPTY_CONTACTS)
                self._last_contacts[env_idx][key] = (positions, forces)
                if forces.size == 0:
                    continue
                self._contacted_pairs[env_idx].add(key)
                self._samples[env_idx].append(
                    {
                        "pair": key,
                        "step": self._step,
                        "positions": positions,
                        "forces": forces,
                        "prev_positions": prev_positions,
                        "prev_forces": prev_forces,
                        "states": dict(self._state_before_step[env_idx]),
                    }
                )
            # Keep the history bounded during long episodes.
            self._samples[env_idx] = self._samples[env_idx][-self.max_samples :]

    def _resolve_entity(self, path):
        path = str(path)
        for root_path, entity in self._roots.items():
            if path == root_path or path.startswith(root_path + "/"):
                return entity
        return None

    def _on_contact_report(self, contact_headers, contact_data):
        if not self.ready:
            return
        from pxr import PhysicsSchemaTools

        for header in contact_headers:
            first = self._resolve_entity(PhysicsSchemaTools.intToSdfPath(header.collider0))
            if first is None and hasattr(header, "actor0"):
                first = self._resolve_entity(PhysicsSchemaTools.intToSdfPath(header.actor0))
            second = self._resolve_entity(PhysicsSchemaTools.intToSdfPath(header.collider1))
            if second is None and hasattr(header, "actor1"):
                second = self._resolve_entity(PhysicsSchemaTools.intToSdfPath(header.actor1))
            if first is None or second is None or first[0] != second[0] or first[1] == second[1]:
                continue
            env_idx = first[0]
            if not self.is_env_ready(env_idx):
                continue
            key = frozenset((first[1], second[1]))
            self._contacted_pairs[env_idx].add(key)
            count = int(getattr(header, "num_contact_data", 0) or 0)
            if count <= 0:
                continue
            origin = self._env_origins[env_idx]
            start = int(getattr(header, "contact_data_offset", 0))
            for contact in contact_data[start : start + count]:
                position = np.asarray(getattr(contact, "position", ()), dtype=float).reshape(-1)
                impulse = np.asarray(getattr(contact, "impulse", ()), dtype=float).reshape(-1)
                if position.size < 3 or impulse.size < 3:
                    continue
                position = position[:3]
                impulse_norm = float(np.linalg.norm(impulse[:3]))
                if not np.all(np.isfinite(position)) or not np.isfinite(impulse_norm):
                    continue
                if origin is not None:
                    position = position - origin
                self._pending_events[env_idx].append((key, position, impulse_norm / self.dt))

    def _on_post_physics_step(self):
        previous = self._previous_post_physics_step_callback
        if previous is not None:
            previous()
        # Attribute contact-report events to the env step being simulated.
        for env_idx, events in enumerate(self._pending_events):
            for key, position, force in events:
                self._step_events[env_idx].setdefault(key, []).append((position, force))
        self._pending_events = [[] for _ in range(self.num_envs)]
