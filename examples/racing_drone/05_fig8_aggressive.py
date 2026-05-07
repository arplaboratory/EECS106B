# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Aggressive dual-drone lemniscate, viewed from a low chase camera.

A repurpose of the gentler ``04_fig8_compare.py`` with three changes:

  - **Wider, faster figure-8** (default radius 4.0 m, period 2.5 s) so the drones
    sweep through tighter banking angles and cross the centre at high speed.
  - **Higher per-drone speed/accel envelopes** so neither lags the reference
    and the trajectory really is a fig-8 rather than a slow ellipse.
  - **Low chase camera** (–y, slightly elevated, looking back at the centre)
    instead of the original isometric overview.

Run inside the EECS106B distrobox container:

    cd /workspace/omni_drones/examples
    PYTHONEXE=/opt/drone_venv/bin/python /workspace/isaacsim/python.sh \\
        play_fig8_aggressive.py --headless --enable_cameras \\
            --width 3840 --height 2160 --fps 60 --frames 720
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Aggressive Iris+FiveIn fig-8.")
parser.add_argument("--width", type=int, default=3840)
parser.add_argument("--height", type=int, default=2160)
parser.add_argument("--frames", type=int, default=720)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--output", type=str, default="demo_fig8_aggressive.mp4")
parser.add_argument("--separation", type=float, default=2.5)
parser.add_argument("--altitude", type=float, default=1.5)
parser.add_argument("--takeoff_seconds", type=float, default=1.5)
parser.add_argument("--fig8_radius", type=float, default=2.0,
                    help="Lemniscate radius (m).")
parser.add_argument("--fig8_period", type=float, default=2.0,
                    help="Seconds per lap. 2.0 s pushes the reference past "
                         "iris's envelope while the 5-inch drone still tracks. "
                         "Lower = more aggressive.")
parser.add_argument("--vertical_amp", type=float, default=0.3,
                    help="Vertical sinusoidal swing amplitude (m).")
parser.add_argument("--cam_pos", type=float, nargs=3, default=(0.0, 0.0, 9.5),
                    help="Camera position (m). Default is a top-down view: "
                         "(0, 0, altitude + ~8) looking straight down so the "
                         "image plane is parallel to the ground and both "
                         "drones stay in frame for the whole lemniscate.")
parser.add_argument("--bitrate", type=str, default="30M",
                    help="ffmpeg target bitrate for the final encode "
                         "(default 30 Mbps = clean 4K).")
parser.add_argument("--encode_crf", type=int, default=16,
                    help="x264 CRF (lower = higher quality, default 16).")
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

from omni_drones.robots.assets.five_in_drone import FIVE_IN_DRONE


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


# Critically-damped PD profiles. omega_n controls response speed:
#  - higher → tighter tracking, more force at corners (more agility)
#  - lower  → more lag, smoother sweeps
# Iris is the heavier multirotor; FiveIn is the lightweight racing drone with
# 2x the agility. Soft max_accel/max_speed caps prevent ICs from teleporting.
DRONE_PROFILES = {
    "iris":    {"omega_n": 4.0, "zeta": 1.0, "yaw_omega_n": 4.0,
                "max_speed":  8.0, "max_accel": 14.0},
    "five_in": {"omega_n": 7.0, "zeta": 1.0, "yaw_omega_n": 7.0,
                "max_speed": 18.0, "max_accel": 35.0},
}


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


# Reuse the known-good 3/4-iso-from-above orientation from
# 04_fig8_compare.py — it points camera +x_world, +y_world, -z_world (i.e.,
# from "right-back-up" toward the origin). Combined with an elevated
# args.cam_pos this keeps the drones in frame.
_CAM_QUAT = (0.5, -0.2, -0.35, 0.77)


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
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=0.53),
    )
    iris = IRIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Iris")
    five_in = FIVE_IN_CFG.replace(prim_path="{ENV_REGEX_NS}/FiveIn")
    cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam",
        update_period=0.0,
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 35 mm gives ~30° HFOV so the lemniscate fills the frame from
            # z≈9.5 m without leaving the drones as tiny dots.
            focal_length=35.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(args.cam_pos),
            rot=_CAM_QUAT,
            convention="world",
        ),
    )


def fig8_pose(t: float, radius: float, alt: float, period: float, vert_amp: float):
    """Bernoulli lemniscate at ``alt`` with optional vertical sinusoidal swing."""
    omega = 2.0 * math.pi / period
    s = omega * t
    denom = 1.0 + math.sin(s) ** 2
    x = radius * math.cos(s) / denom
    y = radius * math.sin(s) * math.cos(s) / denom
    # vertical undulation that hits maxima at the lobe centres
    z = alt + vert_amp * math.sin(2 * s)
    eps = 1e-3
    s_a = omega * (t + eps)
    denom_a = 1.0 + math.sin(s_a) ** 2
    xa = radius * math.cos(s_a) / denom_a
    ya = radius * math.sin(s_a) * math.cos(s_a) / denom_a
    yaw = math.atan2(ya - y, xa - x)
    return (x, y, z), yaw


def reference_pose(t: float):
    if t < args.takeoff_seconds:
        s = t / args.takeoff_seconds
        smoothed = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        return (0.0, 0.0, args.altitude * smoothed), 0.0
    return fig8_pose(
        t - args.takeoff_seconds, args.fig8_radius, args.altitude,
        args.fig8_period, args.vertical_amp,
    )


class PDTracker:
    """Critically-damped 2nd-order pursuer.

    Each axis follows ``ddx = ωn² * (ref - x) - 2 ζ ωn * dx``. With ζ = 1 and
    a per-drone ωn this gives no overshoot, smooth swept corners, and physical
    momentum at the lemniscate crossing — no rate-limit chops, no body-tilt
    flip-flop. Speed and accel are still soft-capped so the drone doesn't
    teleport when far from the reference (e.g. take-off).
    """

    def __init__(self, profile, init_pos, device):
        self.omega_n = profile["omega_n"]                 # rad/s, response BW
        self.zeta = profile.get("zeta", 1.0)
        self.max_speed = profile["max_speed"]
        self.max_accel = profile["max_accel"]
        self.yaw_omega_n = profile["yaw_omega_n"]
        self.pos = torch.tensor(init_pos, device=device, dtype=torch.float32)
        self.vel = torch.zeros(3, device=device)
        self.yaw = torch.tensor(0.0, device=device)
        self.yaw_rate = torch.tensor(0.0, device=device)

    @staticmethod
    def _wrap(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def step(self, ref_pos, ref_yaw, dt):
        ref_pos = torch.as_tensor(ref_pos, device=self.pos.device,
                                  dtype=self.pos.dtype)

        # 2nd-order PD on position.
        wn, z = self.omega_n, self.zeta
        accel = (wn * wn) * (ref_pos - self.pos) - (2.0 * z * wn) * self.vel
        a_mag = accel.norm()
        if a_mag > self.max_accel:
            accel = accel * (self.max_accel / a_mag)
        self.vel = self.vel + accel * dt
        v_mag = self.vel.norm()
        if v_mag > self.max_speed:
            self.vel = self.vel * (self.max_speed / v_mag)
        self.pos = self.pos + self.vel * dt

        # 2nd-order PD on yaw with shortest-arc wrap.
        wny = self.yaw_omega_n
        ref_yaw_t = torch.as_tensor(ref_yaw, device=self.yaw.device,
                                    dtype=self.yaw.dtype)
        err = self._wrap(ref_yaw_t - self.yaw)
        yaw_accel = (wny * wny) * err - (2.0 * wny) * self.yaw_rate
        self.yaw_rate = self.yaw_rate + yaw_accel * dt
        self.yaw = self._wrap(self.yaw + self.yaw_rate * dt)
        return self.pos.clone(), self.vel.clone(), self.yaw.clone()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args.fps, device=args.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = SceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    iris: Articulation = scene["iris"]
    five_in: Articulation = scene["five_in"]
    camera: Camera = scene["cam"]
    sep = args.separation / 2.0
    device = args.device

    # Override the camera offset with a real look-at: the IsaacLab "world"
    # convention quat is fiddly to compute by hand, but
    # ``set_world_poses_from_view(eyes, targets)`` does it correctly.
    # tiny y-offset on the target avoids the gimbal-lock case when the camera
    # sits directly above its target (eye.xy == target.xy makes "up" ambiguous).
    camera.set_world_poses_from_view(
        eyes=torch.tensor([list(args.cam_pos)], device=device, dtype=torch.float32),
        targets=torch.tensor([[0.0, 0.001, args.altitude]], device=device,
                             dtype=torch.float32),
    )

    iris_tracker = PDTracker(DRONE_PROFILES["iris"], (0.0, 0.0, 0.05), device)
    five_tracker = PDTracker(DRONE_PROFILES["five_in"], (0.0, 0.0, 0.05), device)

    # Low-pass smoothed body-frame tilts so the drone leans into the
    # acceleration but doesn't visually flicker each step.
    iris_tilt = {"roll": torch.tensor(0.0, device=device),
                 "pitch": torch.tensor(0.0, device=device)}
    five_tilt = {"roll": torch.tensor(0.0, device=device),
                 "pitch": torch.tensor(0.0, device=device)}
    tilt_alpha = 0.2  # per-step EMA coefficient (~5-frame lag at 60 fps)

    dt = 1.0 / args.fps
    frames = []
    print(f"[aggro] rendering {args.frames} frames {args.width}x{args.height} @ {args.fps}fps "
          f"-> {args.output}  (radius={args.fig8_radius}, period={args.fig8_period})",
          flush=True)

    for i in range(args.frames):
        if sim.is_stopped():
            break
        t = i * dt
        ref_pos, ref_yaw = reference_pose(t)

        iris_pos, iris_vel, iris_yaw = iris_tracker.step(ref_pos, ref_yaw, dt)
        five_pos, five_vel, five_yaw = five_tracker.step(ref_pos, ref_yaw, dt)

        # Body-frame tilt: pitch nose-down by forward velocity in body frame,
        # bank by lateral velocity in body frame. EMA-smoothed so corners
        # don't flicker. A racing drone leans INTO the acceleration vector,
        # but using velocity is a good proxy and avoids needing an accel
        # signal when the tracker is critically damped.
        def _body_tilt(vel_w, yaw_t, k_pitch, k_roll, max_lean):
            cy, sy = torch.cos(yaw_t), torch.sin(yaw_t)
            v_fwd = vel_w[0] * cy + vel_w[1] * sy           # body +x speed
            v_lat = -vel_w[0] * sy + vel_w[1] * cy          # body +y speed
            pitch = torch.clamp(-k_pitch * v_fwd, min=-max_lean, max=max_lean)
            roll = torch.clamp(k_roll * v_lat, min=-max_lean, max=max_lean)
            return roll, pitch

        ir_roll_t, ir_pitch_t = _body_tilt(
            iris_vel, iris_yaw, k_pitch=0.06, k_roll=0.06, max_lean=0.45
        )
        fv_roll_t, fv_pitch_t = _body_tilt(
            five_vel, five_yaw, k_pitch=0.05, k_roll=0.05, max_lean=0.80
        )
        iris_tilt["roll"] = iris_tilt["roll"] + tilt_alpha * (ir_roll_t - iris_tilt["roll"])
        iris_tilt["pitch"] = iris_tilt["pitch"] + tilt_alpha * (ir_pitch_t - iris_tilt["pitch"])
        five_tilt["roll"] = five_tilt["roll"] + tilt_alpha * (fv_roll_t - five_tilt["roll"])
        five_tilt["pitch"] = five_tilt["pitch"] + tilt_alpha * (fv_pitch_t - five_tilt["pitch"])
        iris_roll, iris_pitch = iris_tilt["roll"], iris_tilt["pitch"]
        five_roll, five_pitch = five_tilt["roll"], five_tilt["pitch"]
        iris_quat = quat_from_euler_xyz(
            iris_roll.unsqueeze(0), iris_pitch.unsqueeze(0), iris_yaw.unsqueeze(0)
        )
        five_quat = quat_from_euler_xyz(
            five_roll.unsqueeze(0), five_pitch.unsqueeze(0), five_yaw.unsqueeze(0)
        )

        iris_world_pos = iris_pos.clone().unsqueeze(0)
        iris_world_pos[0, 0] -= sep
        five_world_pos = five_pos.clone().unsqueeze(0)
        five_world_pos[0, 0] += sep

        iris.write_root_pose_to_sim(torch.cat([iris_world_pos, iris_quat], dim=-1))
        five_in.write_root_pose_to_sim(torch.cat([five_world_pos, five_quat], dim=-1))

        if iris.num_joints >= 4:
            spin_iris = torch.tensor([[400.0, -400.0, 400.0, -400.0]], device=device)
            iris.write_joint_state_to_sim(
                position=iris.data.joint_pos,
                velocity=spin_iris[:, : iris.num_joints],
            )
        if five_in.num_joints >= 4:
            spin_five = torch.tensor([[1500.0, -1500.0, 1500.0, -1500.0]], device=device)
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
            print(f"[aggro] step {i:4d}  ref=({ref_pos[0]:+.2f},{ref_pos[1]:+.2f},"
                  f"{ref_pos[2]:+.2f}) yaw={ref_yaw:+.2f}  iris_v={iris_vel.norm():.2f} m/s  "
                  f"five_v={five_vel.norm():.2f} m/s", flush=True)

    out_dir = os.path.dirname(args.output) or "."
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"[aggro] writing {args.output} ({len(frames)} frames) "
          f"crf={args.encode_crf} bitrate={args.bitrate}", flush=True)
    from torchvision.io import write_video
    video = torch.stack(frames).to(torch.uint8)
    write_video(
        args.output,
        video,
        fps=int(args.fps),
        video_codec="libx264",
        options={
            "crf": str(int(args.encode_crf)),
            "preset": "slow",
            "b": str(args.bitrate),
            "pix_fmt": "yuv420p",
            "movflags": "+faststart",
        },
    )
    print(f"[aggro] done -> {os.path.abspath(args.output)}  "
          f"shape={tuple(video.shape)}  mean={video.float().mean().item():.2f}",
          flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
