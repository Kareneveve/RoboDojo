"""Bridge IsaacLab env tensors and PhysX tensors.

``env_cfg/sim/*.yml`` ``device`` is the compute device (``cpu`` or ``cuda:0``).
PhysX may live on the other device when ``overwrite_gpu_setting`` forces CPU
cloth. Reads are copied to the env device; writes are copied back to PhysX.
When PhysX is on GPU, Isaac Sim split velocity writes are routed through
``set_velocities``, and host-side pose APIs convert CUDA tensors before NumPy.
"""

from __future__ import annotations

import numpy as np
import torch

_PATCHED = False

_PHYSX_STATE_PROPERTIES = (
    "joint_pos",
    "joint_vel",
    "root_link_pose_w",
    "root_com_vel_w",
    "body_link_pose_w",
    "body_com_vel_w",
    "body_com_acc_w",
    "body_incoming_joint_wrench_b",
)

_PHYSX_VIEW_WRITES = (
    "set_dof_actuation_forces",
    "set_dof_position_targets",
    "set_dof_velocity_targets",
    "set_dof_positions",
    "set_dof_velocities",
    "set_root_transforms",
    "set_root_velocities",
    "apply_forces_and_torques_at_position",
)

_JOINT_TARGET_SETTERS = {
    "set_joint_position_target": "joint_pos_target",
    "set_joint_velocity_target": "joint_vel_target",
    "set_joint_effort_target": "joint_effort_target",
}


def as_torch_device(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(str(device))


def is_cpu_device(device) -> bool:
    return as_torch_device(device).type == "cpu"


def move_to_device(tensor: torch.Tensor | None, device) -> torch.Tensor | None:
    if tensor is None:
        return None
    target = as_torch_device(device)
    return tensor if tensor.device == target else tensor.to(target)


def as_numpy(value, dtype=None) -> np.ndarray:
    """Copy a tensor or array to host once. Prefer this over per-element ``.item()``."""

    if value is None:
        raise TypeError("as_numpy() expected an array or tensor, got None")
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.device.type != "cpu":
            detached = detached.cpu()
        array = detached.numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        return np.asarray(array, dtype=dtype)
    return array


def _is_cuda_view(view) -> bool:
    device = getattr(view, "_device", None)
    return device is not None and "cuda" in str(device)


def _as_velocity_block(value, count: int, width: int) -> np.ndarray:
    array = as_numpy(value, dtype=np.float32).reshape(-1, width)
    if array.shape[0] == 1 and count > 1:
        array = np.repeat(array, count, axis=0)
    if array.shape[0] != count:
        raise ValueError(f"Velocity block has {array.shape[0]} rows; expected {count}.")
    return array


def _set_gpu_partial_velocities(view, set_velocities, *, linear=None, angular=None, indices=None):
    """Compose a 6D write so GPU PhysX never hits set_linear/angular_velocities."""

    try:
        current = view.get_velocities(indices)
    except Exception:
        current = None
    if current is None:
        count = 1 if indices is None else len(indices)
        current = np.zeros((count, 6), dtype=np.float32)
    else:
        current = as_numpy(current, dtype=np.float32).reshape(-1, 6)
    if linear is not None:
        current[:, :3] = _as_velocity_block(linear, current.shape[0], 3)
    if angular is not None:
        current[:, 3:] = _as_velocity_block(angular, current.shape[0], 3)
    device = getattr(view, "_device", "cuda:0")
    if not getattr(view, "is_physics_handle_valid", lambda: True)():
        return None
    payload = torch.as_tensor(np.ascontiguousarray(current), dtype=torch.float32)
    payload = payload.to(device)
    return set_velocities(view, payload, indices)


def _wrap_gpu_split_velocity_writes(cls) -> None:
    if getattr(cls, "_miverse_gpu_velocity_wrapped", False):
        return
    orig_linear = cls.set_linear_velocities
    orig_angular = cls.set_angular_velocities
    orig_combined = cls.set_velocities

    def set_linear_velocities(self, velocities=None, indices=None):
        if _is_cuda_view(self):
            return _set_gpu_partial_velocities(self, orig_combined, linear=velocities, indices=indices)
        return orig_linear(self, velocities, indices)

    def set_angular_velocities(self, velocities=None, indices=None):
        if _is_cuda_view(self):
            return _set_gpu_partial_velocities(self, orig_combined, angular=velocities, indices=indices)
        return orig_angular(self, velocities, indices)

    cls.set_linear_velocities = set_linear_velocities
    cls.set_angular_velocities = set_angular_velocities
    cls._miverse_gpu_velocity_wrapped = True


def _patch_isaacsim_gpu_velocity_writes() -> None:
    from isaacsim.core.prims.impl.articulation import Articulation
    from isaacsim.core.prims.impl.rigid_prim import RigidPrim

    _wrap_gpu_split_velocity_writes(Articulation)
    _wrap_gpu_split_velocity_writes(RigidPrim)


def _copy_to_device(tensor: torch.Tensor, device, cache: torch.Tensor | None):
    """Reuse ``cache`` for host→device copies so reads do not allocate every step."""

    target = as_torch_device(device)
    if tensor.device == target:
        return tensor, cache
    if cache is None or cache.shape != tensor.shape or cache.dtype != tensor.dtype or cache.device != target:
        cache = tensor.to(target)
    else:
        cache.copy_(tensor)
    return cache, cache


def _move_arg(value, device):
    if isinstance(value, torch.Tensor):
        return move_to_device(value, device)
    return value


def _physx_view_device(view) -> torch.device:
    for getter_name in ("get_dof_positions", "get_dof_velocities"):
        getter = getattr(view, getter_name, None)
        if getter is None:
            continue
        sample = getter()
        if isinstance(sample, torch.Tensor):
            return sample.device
    return torch.device("cpu")


def _sync_previous_joint_vel(data) -> None:
    previous = getattr(data, "_previous_joint_vel", None)
    moved = move_to_device(previous, data.device)
    if moved is not None and moved is not previous:
        data._previous_joint_vel = moved


def _wrap_state_property(cls, name: str) -> None:
    orig_get = getattr(cls, name).fget

    def getter(self):
        value = orig_get(self)
        if not isinstance(value, torch.Tensor):
            return value
        host_attr = f"_miverse_host_{name}"
        moved, cached = _copy_to_device(value, self.device, getattr(self, host_attr, None))
        if cached is not None:
            setattr(self, host_attr, cached)
        if moved is not value:
            isaac_cache = getattr(self, f"_{name}", None)
            if isaac_cache is not None and hasattr(isaac_cache, "data"):
                isaac_cache.data = moved
        if name == "joint_vel":
            _sync_previous_joint_vel(self)
        return moved

    setattr(cls, name, property(getter))


def _wrap_physx_view_writes(view) -> None:
    if view is None or getattr(view, "_miverse_write_wrapped", False):
        return
    physx_device = _physx_view_device(view)
    for name in _PHYSX_VIEW_WRITES:
        orig = getattr(view, name, None)
        if orig is None:
            continue

        def _wrapped(*args, orig=orig, device=physx_device, **kwargs):
            return orig(
                *[_move_arg(arg, device) for arg in args],
                **{key: _move_arg(value, device) for key, value in kwargs.items()},
            )

        setattr(view, name, _wrapped)
    view._miverse_write_wrapped = True


def _patch_articulation_data() -> None:
    from isaaclab.assets.articulation.articulation_data import ArticulationData

    orig_init = ArticulationData.__init__
    orig_update = ArticulationData.update

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _sync_previous_joint_vel(self)

    def update(self, dt: float):
        _sync_previous_joint_vel(self)
        return orig_update(self, dt)

    ArticulationData.__init__ = init
    ArticulationData.update = update
    for name in _PHYSX_STATE_PROPERTIES:
        _wrap_state_property(ArticulationData, name)


def _targets_live_on_cuda(articulation) -> bool:
    tensor = getattr(getattr(articulation, "data", None), "joint_pos_target", None)
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"


def _as_host_index(ids):
    if ids is None:
        return slice(None)
    if isinstance(ids, slice):
        return ids
    return np.asarray(as_numpy(ids), dtype=np.int64).reshape(-1)


def _assign_host_target(buffer: np.ndarray, env_ids, joint_ids, target) -> None:
    env_idx = _as_host_index(env_ids)
    joint_idx = _as_host_index(joint_ids)
    if not isinstance(env_idx, slice) and not isinstance(joint_idx, slice):
        env_idx = env_idx[:, None]
    # NumPy preserves IsaacLab's Cartesian selection and broadcasting, including
    # partial slices and a single joint target shared by multiple environments.
    buffer[env_idx, joint_idx] = as_numpy(target)


def ensure_host_joint_targets(articulation) -> dict[str, np.ndarray] | None:
    """Cache CUDA joint targets on CPU; setters defer uploads until sim writes."""
    host = getattr(articulation, "_miverse_host_joint_targets", None)
    if host is not None:
        return host
    if not _targets_live_on_cuda(articulation):
        return None
    host = {}
    for name in _JOINT_TARGET_SETTERS.values():
        tensor = getattr(articulation.data, name, None)
        if isinstance(tensor, torch.Tensor):
            host[name] = as_numpy(tensor).copy()
    articulation._miverse_host_joint_targets = host
    articulation._miverse_host_targets_dirty = False
    return host


def flush_host_joint_targets(articulation) -> None:
    """Upload each target buffer once after the host control updates."""
    host = getattr(articulation, "_miverse_host_joint_targets", None)
    if host is None or not getattr(articulation, "_miverse_host_targets_dirty", False):
        return
    for name, array in host.items():
        dest = getattr(articulation.data, name, None)
        if isinstance(dest, torch.Tensor):
            dest.copy_(torch.from_numpy(np.ascontiguousarray(array)))
    articulation._miverse_host_targets_dirty = False


def _to_isaac_index(ids, device):
    """IsaacLab compares indices to ``slice(None)`` then does ``ids[:, None]``.

    A multi-element numpy array makes that comparison an array, which cannot be
    used as a bool. Lists pass the comparison but then fail on ``[:, None]``.
    """
    if ids is None or isinstance(ids, slice):
        return ids
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _prepare_isaac_joint_target(target, joint_ids, env_ids, like):
    if not isinstance(like, torch.Tensor):
        return target, joint_ids, env_ids
    if isinstance(target, np.ndarray):
        target = torch.as_tensor(target, dtype=like.dtype, device=like.device)
    elif isinstance(target, torch.Tensor) and (target.device != like.device or target.dtype != like.dtype):
        target = target.to(device=like.device, dtype=like.dtype)
    return (
        target,
        _to_isaac_index(joint_ids, like.device),
        _to_isaac_index(env_ids, like.device),
    )


def _wrap_joint_target_setters(cls) -> None:
    """Defer CUDA target uploads and normalize inputs for native CPU setters."""
    if getattr(cls, "_miverse_joint_target_wrapped", False):
        return
    for method_name, attr in _JOINT_TARGET_SETTERS.items():
        orig = getattr(cls, method_name)

        def _wrapped(self, target, joint_ids=None, env_ids=None, *, orig=orig, attr=attr):
            host = ensure_host_joint_targets(self)
            if host is not None and attr in host:
                _assign_host_target(host[attr], env_ids, joint_ids, target)
                self._miverse_host_targets_dirty = True
                return None
            target, joint_ids, env_ids = _prepare_isaac_joint_target(
                target, joint_ids, env_ids, getattr(self.data, attr)
            )
            return orig(self, target, joint_ids=joint_ids, env_ids=env_ids)

        setattr(cls, method_name, _wrapped)
    cls._miverse_joint_target_wrapped = True


def _patch_articulation_writes() -> None:
    from isaaclab.assets.articulation.articulation import Articulation

    orig_initialize = Articulation._initialize_impl
    orig_write = Articulation.write_data_to_sim
    _wrap_joint_target_setters(Articulation)

    def initialize_impl(self):
        orig_initialize(self)
        _wrap_physx_view_writes(getattr(self, "root_physx_view", None))
        # IsaacLab can recreate target buffers when an articulation is rebound.
        self._miverse_host_joint_targets = None
        ensure_host_joint_targets(self)

    def write_data_to_sim(self):
        flush_host_joint_targets(self)
        _wrap_physx_view_writes(getattr(self, "root_physx_view", None))
        return orig_write(self)

    Articulation._initialize_impl = initialize_impl
    Articulation.write_data_to_sim = write_data_to_sim


def align_isaacsim_view_to_physx(obj) -> None:
    """Point Isaac Sim prim views at the PhysX tensor device.

    Scene objects resolve indices on ``view._device`` (the env device), but
    PhysX transforms stay on CPU when ``overwrite_gpu_setting=0``.
    """
    views = [
        getattr(obj, "_prim_view", None),
        getattr(obj, "_articulation_view", None),
        obj,
    ]
    seen = set()
    for view in views:
        if view is None or id(view) in seen:
            continue
        seen.add(id(view))
        _set_view_device_from_physx(view)


def _set_view_device_from_physx(view) -> None:
    if not hasattr(view, "_device"):
        return
    physics_view = getattr(view, "_physics_view", None)
    sample = None
    if physics_view is not None:
        for name in ("get_transforms", "get_root_transforms", "get_dof_positions"):
            getter = getattr(physics_view, name, None)
            if getter is None:
                continue
            try:
                sample = getter()
            except Exception:
                continue
            if isinstance(sample, torch.Tensor):
                break
    if not isinstance(sample, torch.Tensor):
        return
    device = "cpu" if sample.device.type == "cpu" else str(sample.device)
    view._device = device


def patch_isaaclab_articulation_data() -> None:
    """Install read/write copies between the env device and PhysX."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_articulation_data()
    _patch_articulation_writes()
    _patch_isaacsim_gpu_velocity_writes()
    _PATCHED = True
