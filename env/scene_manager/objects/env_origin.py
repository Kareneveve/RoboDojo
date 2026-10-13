"""Keep object origins on the host and subtract them on the pose device."""

import numpy as np
import torch

from env.environment.isaac.physx_device import as_numpy


def store_env_origin(origin) -> np.ndarray:
    """Return a host xyz copy for a tensor, array, or sequence."""
    if origin is None:
        return np.zeros(3, dtype=np.float64)
    values = as_numpy(origin, dtype=np.float64).reshape(-1)
    if values.size < 3:
        values = np.pad(values, (0, 3 - int(values.size)))
    return np.array(values[:3], dtype=np.float64, copy=True)


def shift_by_origin(pos, origin, *, relative: bool = True, dtype=np.float32) -> np.ndarray:
    """Convert world xyz to the environment frame on the host."""
    pos = as_numpy(pos, dtype=np.float64).reshape(-1)
    result = pos[:3] - store_env_origin(origin) if relative else pos[:3]
    return np.asarray(result, dtype=dtype)


def apply_origin_to_pose_tensor(pose, origin, *, relative: bool = True):
    """Subtract the origin on the pose tensor's device, preserving orientation."""
    if not relative or pose is None or origin is None or not torch.is_tensor(pose) or pose.numel() == 0:
        return pose
    origin_tensor = torch.as_tensor(store_env_origin(origin), device=pose.device, dtype=pose.dtype)
    pose[..., :3] -= origin_tensor
    return pose
