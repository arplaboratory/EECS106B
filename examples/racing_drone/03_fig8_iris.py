# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Iris drone flying a horizontal Lissajous fig-8, viewed from a single fixed
external camera.

Uses the IsaacLab framework with the iris USD as a kinematic-puppet articulation
(``disable_gravity=True`` + ``write_root_pose_to_sim`` each tick). No flight
controller — the drone is driven directly along the analytic trajectory. A
single IsaacLab ``Camera`` watches from a fixed pose; the captured frames are
written to ``play_fig8_iris_out/fig8.mp4``.

Run inside the EECS106B distrobox container:

    cd /workspace/omni_drones/examples
    PYTHONEXE=/opt/drone_venv/bin/python /workspace/isaacsim/python.sh \\
        play_fig8_iris.py --headless --frames 600
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Iris fig-8 with single onboard cam.")
parser.add_argument("--frames", type=int, default=600)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--radius", type=float, default=2.5)
parser.add_argument("--altitude", type=float, default=1.5)
parser.add_argument("--period", type=float, default=6.0)
parser.add_argument("--takeoff_seconds", type=float, default=1.0)
parser.add_argument("--output", type=str, default="play_fig8_iris_out/fig8.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz


_IRIS_USD = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",
        "omni_drones",
        "robots",
        "assets",
        "usd",
        "iris.usd",
    )
)


IRIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Iris",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_IRIS_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={},
)


class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 0.95)),
    )
    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=0.53),
    )
    iris = IRIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Iris")
    cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FixedCam",
        update_period=0.0,
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 35 mm focal at z≈9.5 m above the action gives ~30° HFOV — both
            # the lemniscate and a few metres of margin fit comfortably.
            focal_length=35.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            # Pose is overwritten at runtime by `set_world_poses_from_view`
            # below. The values here just give IsaacLab something to spawn.
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


def fig8_pose(t: float, radius: float, alt: float, period: float):
    """Lemniscate of Bernoulli at altitude ``alt``; yaw points along the path."""
    omega = 2.0 * math.pi / period
    s = omega * t
    denom = 1.0 + math.sin(s) ** 2
    x = radius * math.cos(s) / denom
    y = radius * math.sin(s) * math.cos(s) / denom
    eps = 1e-3
    s_a = omega * (t + eps)
    denom_a = 1.0 + math.sin(s_a) ** 2
    xa = radius * math.cos(s_a) / denom_a
    ya = radius * math.sin(s_a) * math.cos(s_a) / denom_a
    yaw = math.atan2(ya - y, xa - x)
    return (x, y, alt), yaw


def reference_pose(t: float):
    if t < args.takeoff_seconds:
        s = t / args.takeoff_seconds
        smoothed = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        return (0.0, 0.0, args.altitude * smoothed), 0.0
    return fig8_pose(t - args.takeoff_seconds, args.radius, args.altitude, args.period)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args.fps, device=args.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = SceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    iris: Articulation = scene["iris"]
    camera: Camera = scene["cam"]
    device = args.device
    dt = 1.0 / args.fps

    # Top-down POV: camera sits above the action with the image plane parallel
    # to the ground. Tiny y-offset on the target avoids the gimbal-lock case
    # when eye.xy == target.xy makes the "up" direction ambiguous.
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 9.5]], device=device, dtype=torch.float32),
        targets=torch.tensor([[0.0, 0.001, args.altitude]], device=device,
                             dtype=torch.float32),
    )

    frames = []
    print(f"[fig8-iris] rendering {args.frames} frames at "
          f"{args.width}x{args.height} @ {args.fps}fps -> {args.output}", flush=True)

    for i in range(args.frames):
        if sim.is_stopped():
            break
        t = i * dt
        ref_pos, ref_yaw = reference_pose(t)
        pos = torch.tensor([ref_pos], device=device)
        yaw = torch.tensor([ref_yaw], device=device)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=device), torch.zeros(1, device=device), yaw
        )
        iris.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))

        # spin rotors visually
        if iris.num_joints >= 4:
            spin = torch.tensor([[200.0, -200.0, 200.0, -200.0]], device=device)
            iris.write_joint_state_to_sim(
                position=iris.data.joint_pos,
                velocity=spin[:, : iris.num_joints],
            )

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt)

        rgb = camera.data.output["rgb"][0, ..., :3].clone().cpu()
        frames.append(rgb)

        if i % 60 == 0:
            mean = rgb.float().mean().item()
            print(f"[fig8-iris] step {i:4d}  pos=({ref_pos[0]:+.2f}, {ref_pos[1]:+.2f}, "
                  f"{ref_pos[2]:+.2f})  rgb mean={mean:.1f}", flush=True)

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"[fig8-iris] writing {args.output} ({len(frames)} frames)", flush=True)
    from torchvision.io import write_video
    video = torch.stack(frames).to(torch.uint8)
    write_video(args.output, video, fps=int(args.fps))
    final_mean = video.float().mean().item()
    print(f"[fig8-iris] done -> {os.path.abspath(args.output)}  "
          f"shape={tuple(video.shape)}  overall mean={final_mean:.2f}", flush=True)
    if final_mean < 1.0:
        print("[fig8-iris] WARNING: video looks black; renderer didn't produce data.",
              flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
