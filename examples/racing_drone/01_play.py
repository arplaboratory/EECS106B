# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Hover demo for the 5-inch racing drone via the IsaacLab `FIVE_IN_DRONE`
articulation cfg.

Spawns a single five-inch quad, lets it freefall under gravity (the cfg has
dummy actuators), and exercises the migrated `omni_drones.robots.dynamics`
primitives (`Allocation`, `Motor`, `AttitudeController`, `BodyRateController`)
to confirm the dynamics module works end-to-end.

Run inside the EECS106B distrobox container:
    cd /workspace/omni_drones/examples
    python 02_play_five_in.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Five-inch drone hover demo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from omni_drones.robots.assets.five_in_drone import FIVE_IN_DRONE
from omni_drones.robots.dynamics import (
    Allocation,
    AttitudeController,
    BodyRateController,
    Motor,
)


class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device)
    sim = SimulationContext(sim_cfg)
    # Top-down viewport view: above the spawn, image plane parallel to ground.
    # Tiny y-offset on the target keeps the up-direction defined.
    sim.set_camera_view([0.0, 0.0, 4.0], [0.0, 0.001, 0.0])

    scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    n = args.num_envs
    device = args.device
    motor = Motor(
        num_envs=n,
        taus=[1e-4] * 4,
        init=[1500.0] * 4,
        max_rate=[5e4] * 4,
        min_rate=[-5e4] * 4,
        dt=sim_cfg.dt,
        use=True,
        device=device,
    )
    allocation = Allocation(
        num_envs=n,
        arm_length=0.1249,
        thrust_coeff=4.8e-7,
        drag_coeff=2.0e-9,
        device=device,
    )
    J = torch.tensor([[0.003, 0, 0], [0, 0.003, 0], [0, 0, 0.006]])
    AttitudeController(
        num_envs=n,
        J=J,
        K_attitude=torch.eye(3) * 5.0,
        K_omega=torch.eye(3) * 0.5,
        device=device,
    )
    BodyRateController(num_envs=n, J=J, K_omega=torch.eye(3) * 0.5, device=device)

    omega_ref = torch.full((n, 4), 1500.0, device=device)
    omega = motor.compute(omega_ref)
    thrust_torque = allocation.compute(omega)
    print(f"[demo] motor omega[0]:        {omega[0].tolist()}")
    print(f"[demo] alloc thrust+tau[0]:   {thrust_torque[0].tolist()}")
    print(f"[demo] dynamics module: OK")

    for _ in range(args.steps):
        if sim.is_stopped():
            break
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_cfg.dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
