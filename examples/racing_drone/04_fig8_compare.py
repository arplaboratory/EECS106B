# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Side-by-side comparison: Iris vs 5-inch racing drone.

Both drones start at the same takeoff and follow the same figure-8 reference.
Each drone's tracker is rate-limited by its own physical envelope (TWR + arm
length + inertia), so Iris (TWR ~1.6) falls behind on the tight turns while
the 5-inch (TWR ~9.8) keeps the racing line. Renders to a 4K mp4 by default.

Run inside the EECS106B distrobox container:
    # 4K, 600 frames @ 60fps
    python examples/04_fig8_compare.py --headless --output=demo_fig8.mp4

    # quick smoke test
    python examples/04_fig8_compare.py --headless --width=1280 --height=720 --frames=120
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Iris vs FiveIn fig-8 comparison.")
parser.add_argument("--width", type=int, default=3840, help="Render width (default 4K).")
parser.add_argument("--height", type=int, default=2160, help="Render height (default 4K).")
parser.add_argument("--frames", type=int, default=600)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--output", type=str, default="demo_fig8_compare.mp4")
parser.add_argument("--separation", type=float, default=2.5,
                    help="Lateral spacing between drones (m).")
parser.add_argument("--altitude", type=float, default=1.5,
                    help="Cruise altitude after takeoff (m).")
parser.add_argument("--takeoff_seconds", type=float, default=2.5)
parser.add_argument("--fig8_radius", type=float, default=2.0)
parser.add_argument("--fig8_period", type=float, default=2.5,
                    help="Reference fig-8 lap time (s). 2.5 s requires ~5 m/s "
                         "peak speed which exceeds iris's envelope while the "
                         "5-inch racing drone still tracks tightly.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz

from omni_drones.robots.assets.five_in_drone import FIVE_IN_DRONE


# ---- physical envelopes (used to rate-limit each drone's tracker) ----
# numbers come from the YAMLs / URDFs in the repo:
#   iris.yaml: m=1.52, TWR ≈ 1.6 (4 * 8.55e-6 * 838^2 / (1.52*9.81))
#   five_in:   m=0.50, TWR ≈ 9.8 (4 * 4.8e-7 * 5000^2 / (0.50*9.81))
DRONE_PROFILES = {
    "iris": dict(
        # Iris is a heavyweight (1.5 kg) with TWR ≈ 1.6 — its corner accel
        # is dominated by the available thrust headroom above gravity, so it
        # can sustain ~3-4 m/s on a tight lemniscate but no more.
        max_speed=3.5,        # m/s
        max_accel=6.0,        # m/s^2
        max_yaw_rate=2.0,     # rad/s
    ),
    "five_in": dict(
        # 5-inch racing quad: TWR ≈ 9.8, 1/3 the mass — bandwidth limited by
        # the prop spool-up rather than gravity. Easily tracks the same
        # reference at 4-5× the iris envelope.
        max_speed=18.0,
        max_accel=60.0,
        max_yaw_rate=10.0,
    ),
}


_ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "omni_drones", "robots", "assets")
)
_IRIS_USD = os.path.join(_ASSETS, "usd", "iris.usd")


# Both drones spawned as kinematic IsaacLab articulations (gravity off so we
# drive the kinematic root directly via write_root_pose_to_sim).
def _kinematic_cfg(usd_path: str, prim_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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


IRIS_CFG = _kinematic_cfg(_IRIS_USD, "{ENV_REGEX_NS}/Iris")
FIVE_IN_CFG = FIVE_IN_DRONE.replace(
    prim_path="{ENV_REGEX_NS}/FiveIn",
    spawn=FIVE_IN_DRONE.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
    ),
    init_state=FIVE_IN_DRONE.init_state.replace(pos=(0.0, 0.0, 0.05)),
)


class CompareSceneCfg(InteractiveSceneCfg):
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
    five_in = FIVE_IN_CFG.replace(prim_path="{ENV_REGEX_NS}/FiveIn")
    cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CompareCam",
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
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


def fig8_pose(t: float, radius: float, alt: float, period: float):
    """Lemniscate of Bernoulli at altitude ``alt``."""
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
    """Same shared reference for both drones: rise to altitude, then fig-8."""
    if t < args.takeoff_seconds:
        # smooth quintic ease-in to altitude
        s = t / args.takeoff_seconds
        smoothed = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        return (0.0, 0.0, args.altitude * smoothed), 0.0
    return fig8_pose(t - args.takeoff_seconds, args.fig8_radius, args.altitude, args.fig8_period)


class RateLimitedTracker:
    """Each drone advances toward the reference at its own envelope."""

    def __init__(self, profile: dict, init_pos, device):
        self.max_speed = profile["max_speed"]
        self.max_accel = profile["max_accel"]
        self.max_yaw_rate = profile["max_yaw_rate"]
        self.pos = torch.tensor(init_pos, device=device)
        self.vel = torch.zeros(3, device=device)
        self.yaw = torch.tensor(0.0, device=device)

    def step(self, ref_pos, ref_yaw, dt):
        ref_pos = torch.as_tensor(ref_pos, device=self.pos.device)
        # desired velocity is towards the reference, capped at max_speed
        delta = ref_pos - self.pos
        dist = delta.norm()
        if dist > 1e-6:
            v_des = delta / dist * min(self.max_speed, dist / dt)
        else:
            v_des = torch.zeros_like(delta)
        # accel-limit
        dv = v_des - self.vel
        dv_mag = dv.norm()
        max_dv = self.max_accel * dt
        if dv_mag > max_dv:
            dv = dv * (max_dv / dv_mag)
        self.vel = self.vel + dv
        self.pos = self.pos + self.vel * dt
        # yaw-rate-limit
        dyaw = (torch.tensor(ref_yaw, device=self.yaw.device) - self.yaw)
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        max_dyaw = self.max_yaw_rate * dt
        dyaw = torch.clamp(dyaw, -max_dyaw, max_dyaw)
        self.yaw = self.yaw + dyaw
        return self.pos.clone(), self.vel.clone(), self.yaw.clone()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args.fps, device=args.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = CompareSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    iris: Articulation = scene["iris"]
    five_in: Articulation = scene["five_in"]
    camera: Camera = scene["cam"]
    sep = args.separation / 2.0
    device = args.device
    # point the camera at the action area
    # Top-down POV: camera above the action with image plane parallel to the
    # ground. Tiny y-offset on the target avoids the gimbal-lock case when
    # eye.xy == target.xy makes "up" ambiguous.
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 9.5]], device=device, dtype=torch.float32),
        targets=torch.tensor([[0.0, 0.001, args.altitude]], device=device,
                             dtype=torch.float32),
    )

    iris_tracker = RateLimitedTracker(DRONE_PROFILES["iris"],
                                      init_pos=(0.0, 0.0, 0.05), device=device)
    five_tracker = RateLimitedTracker(DRONE_PROFILES["five_in"],
                                      init_pos=(0.0, 0.0, 0.05), device=device)

    dt = 1.0 / args.fps
    frames = []
    print(f"[demo] rendering {args.frames} frames at {args.width}x{args.height} @ {args.fps}fps "
          f"-> {args.output}", flush=True)

    for i in range(args.frames):
        if sim.is_stopped():
            break
        t = i * dt
        ref_pos, ref_yaw = reference_pose(t)

        iris_pos, iris_vel, iris_yaw = iris_tracker.step(ref_pos, ref_yaw, dt)
        five_pos, five_vel, five_yaw = five_tracker.step(ref_pos, ref_yaw, dt)

        # tilt visualization: tilt forward proportional to forward-accel
        iris_pitch = -torch.clamp(iris_vel.norm() * 0.08, max=0.6)
        five_pitch = -torch.clamp(five_vel.norm() * 0.06, max=0.6)
        iris_quat = quat_from_euler_xyz(
            torch.zeros(1, device=device), iris_pitch.unsqueeze(0), iris_yaw.unsqueeze(0)
        )
        five_quat = quat_from_euler_xyz(
            torch.zeros(1, device=device), five_pitch.unsqueeze(0), five_yaw.unsqueeze(0)
        )

        iris_world_pos = iris_pos.clone().unsqueeze(0)
        iris_world_pos[0, 0] -= sep
        five_world_pos = five_pos.clone().unsqueeze(0)
        five_world_pos[0, 0] += sep

        iris.write_root_pose_to_sim(torch.cat([iris_world_pos, iris_quat], dim=-1))
        five_in.write_root_pose_to_sim(torch.cat([five_world_pos, five_quat], dim=-1))

        # spin rotors visually if the articulation has the joints
        if iris.num_joints >= 4:
            spin_iris = torch.tensor([[200.0, -200.0, 200.0, -200.0]], device=device)
            iris.write_joint_state_to_sim(
                position=iris.data.joint_pos,
                velocity=spin_iris[:, : iris.num_joints],
            )
        if five_in.num_joints >= 4:
            spin_five = torch.tensor([[800.0, -800.0, 800.0, -800.0]], device=device)
            five_in.write_joint_state_to_sim(
                position=five_in.data.joint_pos,
                velocity=spin_five[:, : five_in.num_joints],
            )

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt)

        rgb = camera.data.output["rgb"][0, ..., :3].clone().cpu()
        frames.append(rgb)

        if i % 60 == 0:
            print(f"[demo] step {i:4d}  iris z={iris_pos[2].item():+.2f}  "
                  f"five_in z={five_pos[2].item():+.2f}  "
                  f"|iris-ref|={(iris_pos - torch.as_tensor(ref_pos, device=device)).norm():.2f}  "
                  f"|five-ref|={(five_pos - torch.as_tensor(ref_pos, device=device)).norm():.2f}",
                  flush=True)

    print(f"[demo] writing {args.output} ({len(frames)} frames)", flush=True)
    from torchvision.io import write_video
    video = torch.stack(frames).to(torch.uint8)
    write_video(args.output, video, fps=args.fps)
    print(f"[demo] done -> {os.path.abspath(args.output)}")
    simulation_app.close()


if __name__ == "__main__":
    main()
