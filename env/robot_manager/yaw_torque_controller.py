"""Reusable one-step yaw-rate tracking through a body-frame torque command."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class YawTorqueVelocityConfig:
    """Parameters for discrete one-step (deadbeat) yaw-rate control."""

    effective_inertia_kg_m2: float = 8.65
    max_torque_nm: float = 7500.0
    stop_velocity_threshold_rad_s: float = 0.01
    disturbance_observer_gain: float = 1.0
    max_disturbance_nm: float = 1000.0
    heading_hold_stiffness_nm_per_rad: float = 800.0
    heading_hold_damping_nm_s_per_rad: float = 120.0

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any] | None
    ) -> YawTorqueVelocityConfig:
        values = config or {}
        result = cls(
            effective_inertia_kg_m2=float(
                values.get(
                    "effective_inertia_kg_m2",
                    cls.effective_inertia_kg_m2,
                )
            ),
            max_torque_nm=float(values.get("max_torque_nm", cls.max_torque_nm)),
            stop_velocity_threshold_rad_s=float(
                values.get(
                    "stop_velocity_threshold_rad_s",
                    cls.stop_velocity_threshold_rad_s,
                )
            ),
            disturbance_observer_gain=float(
                values.get(
                    "disturbance_observer_gain",
                    cls.disturbance_observer_gain,
                )
            ),
            max_disturbance_nm=float(
                values.get("max_disturbance_nm", cls.max_disturbance_nm)
            ),
            heading_hold_stiffness_nm_per_rad=float(
                values.get(
                    "heading_hold_stiffness_nm_per_rad",
                    cls.heading_hold_stiffness_nm_per_rad,
                )
            ),
            heading_hold_damping_nm_s_per_rad=float(
                values.get(
                    "heading_hold_damping_nm_s_per_rad",
                    cls.heading_hold_damping_nm_s_per_rad,
                )
            ),
        )
        result.validate()
        return result

    @property
    def heading_hold_enabled(self) -> bool:
        return self.heading_hold_stiffness_nm_per_rad > 0.0

    def validate(self) -> None:
        positive_parameters = {
            "effective_inertia_kg_m2": self.effective_inertia_kg_m2,
            "max_torque_nm": self.max_torque_nm,
            "stop_velocity_threshold_rad_s": self.stop_velocity_threshold_rad_s,
            "max_disturbance_nm": self.max_disturbance_nm,
        }
        invalid = [
            name
            for name, value in positive_parameters.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(
                "yaw-torque velocity parameters must be positive and finite: "
                + ", ".join(invalid)
            )
        if (
            not math.isfinite(self.disturbance_observer_gain)
            or not 0.0 <= self.disturbance_observer_gain <= 1.0
        ):
            raise ValueError("disturbance_observer_gain must be in [0, 1].")
        hold_parameters = {
            "heading_hold_stiffness_nm_per_rad": (
                self.heading_hold_stiffness_nm_per_rad
            ),
            "heading_hold_damping_nm_s_per_rad": (
                self.heading_hold_damping_nm_s_per_rad
            ),
        }
        invalid_hold = [
            name
            for name, value in hold_parameters.items()
            if not math.isfinite(value) or value < 0.0
        ]
        if invalid_hold:
            raise ValueError(
                "heading-hold parameters must be finite and non-negative: "
                + ", ".join(invalid_hold)
            )


class YawTorqueVelocityController:
    """Drive measured yaw speed to the command in one discrete physics step.

    The model is ``I * delta_omega / dt = torque + disturbance``.  The
    disturbance observer uses the previous command and measured response to
    estimate ground/contact resistance, then cancels that estimate on the next
    step.  Torque saturation remains the hard physical safety boundary.

    With PhysX's semi-implicit step, reaching ``target_velocity`` also makes the
    next yaw approximately ``current_yaw + target_velocity * dt``. A zero
    command while the body is still rotating produces one braking
    command.  A zero yaw-rate command switches to heading-hold PD: if a
    target yaw is provided it is tracked softly, otherwise the measured yaw
    is latched on entry.  Hold does not require the body to already be
    still; the damping term brakes residual spin.  Set
    ``heading_hold_stiffness_nm_per_rad`` to 0 to restore clear-on-idle.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        device,
        config: YawTorqueVelocityConfig,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("yaw-torque controller num_envs must be positive.")
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("yaw-torque controller dt must be positive and finite.")
        config.validate()
        self.dt = float(dt)
        self.config = config
        shape = (int(num_envs),)
        self.last_torque = torch.zeros(shape, dtype=torch.float32, device=device)
        self.last_unsaturated_torque = torch.zeros_like(self.last_torque)
        self.last_saturated = torch.zeros(shape, dtype=torch.bool, device=device)
        self.estimated_disturbance = torch.zeros_like(self.last_torque)
        self._previous_velocity = torch.zeros_like(self.last_torque)
        self._has_response_sample = torch.zeros(shape, dtype=torch.bool, device=device)
        self._holding = torch.zeros(shape, dtype=torch.bool, device=device)
        self._hold_yaw = torch.zeros(shape, dtype=torch.float32, device=device)

    def reset(self, env_ids: torch.Tensor, current_yaw: torch.Tensor) -> None:
        """Clear command history; ``current_yaw`` is accepted for adapter API parity."""

        del current_yaw
        indices = env_ids.to(dtype=torch.long)
        self._clear(indices)

    def compute(
        self,
        *,
        env_ids: torch.Tensor,
        target_velocity: torch.Tensor,
        current_yaw: torch.Tensor,
        current_velocity: torch.Tensor,
        target_yaw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return local-Z torque for the selected environments."""

        indices = env_ids.to(dtype=torch.long)
        target_velocity = target_velocity.to(
            dtype=self.last_torque.dtype, device=self.last_torque.device
        )
        current_velocity = current_velocity.to(
            dtype=self.last_torque.dtype, device=self.last_torque.device
        )
        current_yaw = current_yaw.to(
            dtype=self.last_torque.dtype, device=self.last_torque.device
        )
        if target_yaw is not None:
            target_yaw = target_yaw.to(
                dtype=self.last_torque.dtype, device=self.last_torque.device
            )

        stopped = target_velocity == 0.0
        idle = stopped & (
            current_velocity.abs() <= self.config.stop_velocity_threshold_rad_s
        )
        was_holding = self._holding[indices]
        hold_enabled = self.config.heading_hold_enabled
        hold_now = stopped if hold_enabled else idle
        update_observer = self._has_response_sample[indices] & ~was_holding
        observed_disturbance = (
            self.config.effective_inertia_kg_m2
            * (current_velocity - self._previous_velocity[indices])
            / self.dt
            - self.last_torque[indices]
        ).clamp(
            -self.config.max_disturbance_nm,
            self.config.max_disturbance_nm,
        )
        gain = self.config.disturbance_observer_gain
        filtered_disturbance = (1.0 - gain) * self.estimated_disturbance[
            indices
        ] + gain * observed_disturbance
        disturbance = torch.where(
            update_observer,
            filtered_disturbance,
            self.estimated_disturbance[indices],
        )
        self.estimated_disturbance[indices] = disturbance

        unsaturated_torque = (
            self.config.effective_inertia_kg_m2
            * (target_velocity - current_velocity)
            / self.dt
            - disturbance
        )
        if hold_enabled:
            if target_yaw is not None:
                self._hold_yaw[indices] = torch.where(
                    hold_now,
                    target_yaw,
                    self._hold_yaw[indices],
                )
            else:
                entering_hold = hold_now & ~was_holding
                self._hold_yaw[indices] = torch.where(
                    entering_hold,
                    current_yaw,
                    self._hold_yaw[indices],
                )
            heading_error = _wrap_to_pi(current_yaw - self._hold_yaw[indices])
            hold_torque = (
                -self.config.heading_hold_stiffness_nm_per_rad * heading_error
                - self.config.heading_hold_damping_nm_s_per_rad * current_velocity
                - disturbance
            )
            unsaturated_torque = torch.where(
                hold_now, hold_torque, unsaturated_torque
            )
            self._holding[indices] = hold_now
        else:
            unsaturated_torque = torch.where(
                idle,
                torch.zeros_like(unsaturated_torque),
                unsaturated_torque,
            )
            self._holding[indices] = False
        torque = unsaturated_torque.clamp(
            -self.config.max_torque_nm,
            self.config.max_torque_nm,
        )

        self.last_unsaturated_torque[indices] = unsaturated_torque
        self.last_saturated[indices] = torque != unsaturated_torque
        self.last_torque[indices] = torque
        self._previous_velocity[indices] = current_velocity
        self._has_response_sample[indices] = True
        if (not hold_enabled) and torch.any(idle):
            self._clear(indices[idle])
        return torque

    def _clear(self, indices: torch.Tensor) -> None:
        self.last_torque[indices] = 0.0
        self.last_unsaturated_torque[indices] = 0.0
        self.last_saturated[indices] = False
        self.estimated_disturbance[indices] = 0.0
        self._previous_velocity[indices] = 0.0
        self._has_response_sample[indices] = False
        self._holding[indices] = False
        self._hold_yaw[indices] = 0.0


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))
