# Copyright (c) 2025, C.K. Wolfe
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Control-allocation matrix for quadrotor thrust/torque computation.

Maps individual rotor angular velocities to the total body-frame thrust and
torque vector using a standard X-configuration allocation matrix.
"""

import torch

from ._device import default_sim_device


class Allocation:
    """Batched control-allocation for a quadrotor in X-configuration.

    Builds a ``(num_envs, 4, 4)`` allocation matrix that transforms per-rotor
    thrust values into ``[total_thrust, tau_x, tau_y, tau_z]``.

    Attributes:
        _allocation_matrix: Pre-computed allocation matrices for all envs.
        _thrust_coeff: Rotor thrust coefficient (force = coeff * omega^2).
    """

    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device=None, dtype=torch.float32):
        """Initialise the allocation matrix for all environments.

        Args:
            num_envs: Number of parallel environments.
            arm_length: Allocation parameter; planar moment arm is ``arm_length/√2``
                (X-configuration; matches flightstack ``drone_physics.arm_length``).
            thrust_coeff: Rotor thrust constant  (force = coeff * omega^2).
            drag_coeff: Rotor reactive-torque constant.
            device: Torch device; default ``cuda:0`` when available, else ``cpu``.
            dtype: Desired tensor data type.
        """
        if device is None:
            device = default_sim_device()
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    def compute(self, omega):
        """Compute total thrust and body torques from rotor angular velocities.

        Args:
            omega: Rotor angular velocities of shape ``(num_envs, 4)``.

        Returns:
            Tensor of shape ``(num_envs, 4)`` containing
            ``[total_thrust, tau_x, tau_y, tau_z]``.
        """
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque
