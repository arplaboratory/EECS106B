# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Hover the 5-inch racing drone via the IsaacLab `FIVE_IN_DRONE` articulation.

Applies an external thrust + body-frame torque to ``base_link`` every step. The
thrust comes from a P+D controller on altitude (target = 1.0 m). The torque
comes from the existing :class:`AttitudeController` driving ``rpy=(0,0,0)``.

After the loop, the drone's trajectory and altitude history are saved to
``demo_five_in_hover.png`` for visual inspection.

Run inside the EECS106B distrobox container:

    cd /workspace/omni_drones/examples
    PYTHONEXE=/opt/drone_venv/bin/python /workspace/isaacsim/python.sh \\
        03_five_in_hover.py --headless --steps=600
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hover the 5-inch racing drone.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument(
    "--target_altitude",
    type=float,
    default=1.0,
    help="Hover altitude in metres (drone spawns just above ground).",
)
parser.add_argument(
    "--start_altitude",
    type=float,
    default=0.05,
    help="Initial spawn altitude.",
)
parser.add_argument(
    "--plot_path",
    type=str,
    default="demo_five_in_hover.png",
    help="Where to save the trajectory + altitude plot.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from omni_drones.robots.assets.five_in_drone import FIVE_IN_DRONE
from omni_drones.robots.dynamics import (
    Allocation,
    AttitudeController,
    Motor,
)


# Drone parameters (must match the URDF + racing-drone.md).
MASS = 0.5
INERTIA_DIAG = (3.0e-3, 3.0e-3, 6.0e-3)
ARM_LENGTH = 0.1249
THRUST_COEFF = 4.8e-7
DRAG_COEFF = 2.0e-9
GRAVITY = 9.81
HOVER_THRUST = MASS * GRAVITY  # ≈ 4.905 N
OMEGA_HOVER = math.sqrt(HOVER_THRUST / (4.0 * THRUST_COEFF))  # ≈ 1599 rad/s
OMEGA_MAX = 5000.0
KP_ALT = 6.0
KD_ALT = 3.5


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
    # Top-down viewport view: above the hover target, image plane parallel
    # to ground. Tiny y-offset on the target keeps the up-direction defined.
    sim.set_camera_view([0.0, 0.0, args.target_altitude + 4.0],
                        [0.0, 0.001, args.target_altitude])

    scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    n = args.num_envs
    device = args.device
    robot = scene["robot"]

    init_root = robot.data.default_root_state.clone()
    init_root[:, :3] += scene.env_origins
    init_root[:, 2] = args.start_altitude
    robot.write_root_pose_to_sim(init_root[:, :7])
    robot.write_root_velocity_to_sim(init_root[:, 7:])

    motor = Motor(
        num_envs=n,
        taus=[1e-2] * 4,
        init=[OMEGA_HOVER] * 4,
        max_rate=[5e4] * 4,
        min_rate=[-5e4] * 4,
        dt=sim_cfg.dt,
        use=True,
        device=device,
    )
    allocation = Allocation(
        num_envs=n,
        arm_length=ARM_LENGTH,
        thrust_coeff=THRUST_COEFF,
        drag_coeff=DRAG_COEFF,
        device=device,
    )
    J = torch.diag(torch.tensor(INERTIA_DIAG, device=device))
    att_ctrl = AttitudeController(
        num_envs=n,
        J=J,
        K_attitude=torch.eye(3, device=device) * 5.0,
        K_omega=torch.eye(3, device=device) * 0.5,
        device=device,
    )
    A_inv = torch.linalg.inv(allocation._allocation_matrix[0])
    target_attitude = torch.zeros(n, 3, device=device)

    pos_history = torch.zeros(args.steps, n, 3, device="cpu")
    vel_history = torch.zeros(args.steps, n, 3, device="cpu")
    thrust_history = torch.zeros(args.steps, n, device="cpu")

    print(
        f"[hover] omega_hover = {OMEGA_HOVER:.1f} rad/s, "
        f"hover_thrust = {HOVER_THRUST:.3f} N",
        flush=True,
    )

    for step in range(args.steps):
        if sim.is_stopped():
            break

        pos = robot.data.root_pos_w - scene.env_origins
        vel = robot.data.root_lin_vel_w
        quat = robot.data.root_quat_w
        ang_vel = robot.data.root_ang_vel_b

        alt_err = args.target_altitude - pos[:, 2]
        thrust_total = HOVER_THRUST + KP_ALT * alt_err - KD_ALT * vel[:, 2]
        thrust_total = thrust_total.clamp(min=0.0, max=4.0 * THRUST_COEFF * OMEGA_MAX**2)
        torques_body = att_ctrl.compute_moment(target_attitude, quat, ang_vel)

        wrench = torch.cat([thrust_total.unsqueeze(-1), torques_body], dim=-1)
        per_rotor = (A_inv @ wrench.unsqueeze(-1)).squeeze(-1).clamp(min=0.0)
        omega_ref = torch.sqrt(per_rotor / THRUST_COEFF).clamp(min=0.0, max=OMEGA_MAX)
        omega = motor.compute(omega_ref)
        thrust_torque = allocation.compute(omega)
        thrust_actual = thrust_torque[:, 0]
        torques_actual = thrust_torque[:, 1:4]

        forces = torch.zeros(n, 1, 3, device=device)
        forces[:, 0, 2] = thrust_actual
        torques = torques_actual.unsqueeze(1)
        robot.set_external_force_and_torque(forces, torques, body_ids=[0])

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_cfg.dt)

        pos_history[step] = pos.cpu()
        vel_history[step] = vel.cpu()
        thrust_history[step] = thrust_actual.cpu()

        if step % 50 == 0:
            print(
                f"[hover] step={step:4d}  z={pos[0, 2].item():+.3f} m  "
                f"vz={vel[0, 2].item():+.3f} m/s  thrust={thrust_actual[0].item():+.3f} N",
                flush=True,
            )

    final_pos = pos_history[-1, 0]
    print(
        f"[hover] DONE — final pos = ({final_pos[0]:+.3f}, "
        f"{final_pos[1]:+.3f}, {final_pos[2]:+.3f}); "
        f"altitude error = {(final_pos[2] - args.target_altitude):+.3f} m",
        flush=True,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = torch.arange(args.steps).float() * sim_cfg.dt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        for i in range(min(n, 4)):
            ax.plot(t.numpy(), pos_history[:, i, 2].numpy(), label=f"env{i}")
        ax.axhline(args.target_altitude, color="k", linestyle="--", alpha=0.4, label="target")
        ax.set_xlabel("t [s]"); ax.set_ylabel("z [m]"); ax.set_title("Altitude")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)

        ax = axes[1]
        for i in range(min(n, 4)):
            ax.plot(t.numpy(), vel_history[:, i, 2].numpy(), label=f"env{i}")
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.4)
        ax.set_xlabel("t [s]"); ax.set_ylabel("vz [m/s]"); ax.set_title("Vertical velocity")
        ax.legend(loc="upper right"); ax.grid(alpha=0.3)

        ax = axes[2]
        for i in range(min(n, 4)):
            ax.plot(t.numpy(), thrust_history[:, i].numpy(), label=f"env{i}")
        ax.axhline(HOVER_THRUST, color="k", linestyle="--", alpha=0.4, label="m·g")
        ax.set_xlabel("t [s]"); ax.set_ylabel("thrust [N]"); ax.set_title("Total thrust")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(args.plot_path, dpi=120)
        print(f"[hover] wrote {args.plot_path}", flush=True)
    except ImportError:
        print("[hover] matplotlib not available; skipping plot", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
