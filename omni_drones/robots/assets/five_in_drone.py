# Copyright (c) 2025, C.K. Wolfe
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""5-inch freestyle quadcopter :class:`~isaaclab.assets.ArticulationCfg`.

On-disk bundle: ``omni_drones/robots/assets/drones/five_in/`` (``5_in_drone.usd``,
``urdf/5_in_drone.urdf``). Uniform scale ``(1,1,1)``.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

FIVE_IN_DRONE = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.abspath(
            os.path.join(_ASSETS_DIR, "drones", "five_in", "5_in_drone.usd")
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={
            "rotor_0_joint": 200.0,
            "rotor_1_joint": -200.0,
            "rotor_2_joint": 200.0,
            "rotor_3_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
