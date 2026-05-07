# Copyright (c) 2025, C.K. Wolfe
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""First-order motor response model for quadrotor rotors.

Simulates per-rotor angular-velocity dynamics with configurable time constants
and rate limits, integrated via forward-Euler at a fixed time step.
"""

import torch

from ._device import default_sim_device


class Motor:
    """Batched first-order motor model with rate limiting.

    Each rotor tracks a reference angular velocity through a first-order lag
    (time constant ``tau``) whose rate of change is clamped between
    ``min_rate`` and ``max_rate``.

    Attributes:
        omega: Current rotor angular velocities ``(num_envs, num_motors)``.
    """

    def __init__(self, num_envs, taus, init, max_rate, min_rate, dt, use, device=None, dtype=torch.float32):
        """Initialise the motor model.

        Args:
            num_envs: Number of parallel environments.
            taus: Time constant per motor, shape ``(4,)``.
            init: Initial angular velocity per motor in rad/s, shape ``(4,)``.
            max_rate: Upper rate-of-change limit in rad/s^2, shape ``(4,)``.
            min_rate: Lower rate-of-change limit in rad/s^2, shape ``(4,)``.
            dt: Integration time step in seconds.
            use: If ``False``, motor dynamics are bypassed and omega
                tracks the reference instantaneously.
            device: Torch device; default ``cuda:0`` when available, else ``cpu``.
            dtype: Desired tensor data type.
        """
        if device is None:
            device = default_sim_device()
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)

    def compute(self, omega_ref):
        """Advance the motor state by one time step.

        Args:
            omega_ref: Reference angular velocities of shape
                ``(num_envs, num_motors)``.

        Returns:
            Updated angular velocities of shape ``(num_envs, num_motors)``.
        """
        if not self.use:
            self.omega = omega_ref
            return self.omega

        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids):
        """Reset selected environments to their initial motor speeds.

        Args:
            env_ids: Tensor or sequence of environment indices to reset.
        """
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(len(env_ids), -1)
