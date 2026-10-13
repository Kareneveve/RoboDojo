"""Apply MiVerse evaluation velocities to the M92UW floating root."""

from __future__ import annotations

import numpy as np
import torch

from env.robot_manager.yaw_torque_controller import YawTorqueVelocityConfig, YawTorqueVelocityController


class M92UWMobileBaseAdapter:
    """Use root linear velocity for translation and body torque for yaw."""

    base_body_name = "base_footprint"
    # Logical velocity channel consumed by NavEval; this is not a USD joint.
    yaw_joint_name = "root_revolute"
    yaw_velocity_joint_name = yaw_joint_name
    velocity_only_joint_names = frozenset((yaw_joint_name,))

    def __init__(self, *, articulation, dt: float, device, config=None) -> None:
        self.articulation = articulation
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("mobile-base adapter dt must be positive and finite.")
        self.device = device
        self.config = config or {}
        body_ids, body_names = articulation.find_bodies(f"^{self.base_body_name}$", preserve_order=True)
        if len(body_ids) != 1 or body_names != [self.base_body_name]:
            raise RuntimeError(
                f"M92UW torque body must resolve exactly once as {self.base_body_name!r}, got {body_names}."
            )
        self._base_body_id = int(body_ids[0])
        num_envs = int(articulation.num_instances)
        self._yaw_controller = YawTorqueVelocityController(
            num_envs=num_envs,
            dt=self.dt,
            device=self.device,
            config=YawTorqueVelocityConfig.from_mapping(self.config.get("yaw_torque_control", {})),
        )
        self._target_velocities = {env_id: 0.0 for env_id in range(num_envs)}
        self.reset(range(num_envs))

    def read_state(self, env_id: int) -> np.ndarray:
        """Return world-frame root-link pose (7) and velocity (6)."""

        state = self.articulation.data.root_link_state_w[int(env_id)]
        return np.asarray(state.detach().cpu(), dtype=np.float64).copy()

    def apply_velocity(self, env_ids, velocities, target_yaws=None) -> None:
        env_id_list = [int(env_id) for env_id in env_ids]
        velocity_array = np.asarray(velocities, dtype=np.float64)
        if not env_id_list:
            if velocity_array.size:
                raise ValueError("velocities must be empty when env_ids is empty.")
            return
        if velocity_array.shape != (len(env_id_list), 6) or not np.all(np.isfinite(velocity_array)):
            raise ValueError(
                f"mobile-base velocities must be finite shape ({len(env_id_list)}, 6), got {velocity_array.shape}."
            )
        env_ids_tensor = torch.as_tensor(env_id_list, dtype=torch.int32, device=self.device)
        row_ids = env_ids_tensor.to(dtype=torch.long)
        command = torch.as_tensor(velocity_array, dtype=torch.float32, device=self.device)
        target_yaw_tensor = None
        if target_yaws is not None:
            target_yaw_array = np.asarray(target_yaws, dtype=np.float64).reshape(-1)
            if target_yaw_array.shape != (len(env_id_list),) or not np.all(np.isfinite(target_yaw_array)):
                raise ValueError(
                    f"target_yaws must be finite shape ({len(env_id_list)},), got {target_yaw_array.shape}."
                )
            target_yaw_tensor = torch.as_tensor(target_yaw_array, dtype=torch.float32, device=self.device)

        # The root setter takes all six components. Keep measured angular
        # velocity so it does not bypass torque-driven rotation.
        root_velocity = self.articulation.data.root_link_vel_w.index_select(0, row_ids).clone()
        root_velocity[:, :3] = command[:, :3]
        self.articulation.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids_tensor)
        yaw, yaw_velocity = self._read_yaw_state(env_ids_tensor)
        torque = self._yaw_controller.compute(
            env_ids=env_ids_tensor,
            target_velocity=command[:, 5],
            current_yaw=yaw,
            current_velocity=yaw_velocity,
            target_yaw=target_yaw_tensor,
        )
        self._write_yaw_torque(env_ids_tensor, torque)
        for row, env_id in enumerate(env_id_list):
            self._target_velocities[env_id] = float(velocity_array[row, 5])

    def reset(self, env_ids) -> None:
        """Clear selected environments' torque, hold and observer history."""

        env_id_list = [int(env_id) for env_id in env_ids]
        if not env_id_list:
            return
        env_ids_tensor = torch.as_tensor(env_id_list, dtype=torch.int32, device=self.device)
        yaw, _ = self._read_yaw_state(env_ids_tensor)
        self._yaw_controller.reset(env_ids_tensor, yaw)
        self._write_yaw_torque(env_ids_tensor, torch.zeros_like(yaw))
        for env_id in env_id_list:
            self._target_velocities[env_id] = 0.0

    def get_joint_velocity_targets(self, env_id: int) -> dict[str, float]:
        return {self.yaw_joint_name: float(self._target_velocities.get(int(env_id), 0.0))}

    def _read_yaw_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_ids = env_ids.to(dtype=torch.long)
        quaternion = self.articulation.data.body_link_quat_w[row_ids, self._base_body_id]
        angular_velocity_w = self.articulation.data.body_ang_vel_w[row_ids, self._base_body_id]
        yaw = _yaw_from_wxyz_tensor(quaternion)
        local_z_w = _local_z_axis_w(quaternion)
        yaw_velocity = torch.sum(angular_velocity_w * local_z_w, dim=-1)
        return yaw, yaw_velocity

    def _write_yaw_torque(self, env_ids: torch.Tensor, torque_z: torch.Tensor) -> None:
        torques = torch.zeros((len(env_ids), 1, 3), dtype=torch.float32, device=self.device)
        torques[:, 0, 2] = torque_z
        self.articulation.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros_like(torques),
            torques=torques,
            body_ids=[self._base_body_id],
            env_ids=env_ids,
            is_global=False,
        )


def _yaw_from_wxyz_tensor(quaternion: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _local_z_axis_w(quaternion: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
        dim=-1,
    )
