# ----------------------------------------------------------------------------
# Copyright (c) 2026 C. K. Wolfe. All rights reserved.
# NOT FREE TO USE.
# ----------------------------------------------------------------------------
"""Play a trained PPO checkpoint through the DroneRace track and record one
4K POV video per perception-degradation config.

Direct env.step loop (no SyncDataCollector) so the iteration is unambiguous.
At each step we read the drone + gate poses out of the env, render the
geometric segmask via GeometricMask, composite a 4K frame with overlay text,
and append to the current lap's video.  When `steps_per_lap` is hit (or the
real lap_completed flag fires), we close the file and rotate to the next
degradation profile.

Invocation (inside the omnidrones container):

    python scripts/play_pov_segmask.py \\
        task=DroneRace algo=DroneRace \\
        algo.checkpoint_path=/workspace/omni_drones/scripts/wandb/goat-run-3-512-13.30/files/final_model.pt \\
        task.env.num_envs=1 \\
        headless=true \\
        +video_dir=/tmp/segmask_pov \\
        +steps_per_lap=600
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType

from omni_drones import init_simulation_app
from omni_drones.utils.torchrl.transforms import (
    AttitudeController,
    RateController,
    ravel_composite,
)
from omni_drones.learning import ALGOS

FILE_PATH = os.path.dirname(__file__)


def _degrade_mask(
    mask: torch.Tensor,    # (N, H, W) uint8 in {0, 255}
    pixelation_factor: int = 1,
    bernoulli_p: float = 0.0,
    erosion_px: int = 0,
    motion_blur_px: float = 0.0,
) -> torch.Tensor:
    """Onboard-camera-style perception degradations applied to the binary
    gate mask: blocky pixelation, per-pixel Bernoulli flips, binary erosion,
    horizontal motion blur. All knobs independent; pass 0/1 for off."""
    import torch.nn.functional as F
    N, H, W = mask.shape
    if pixelation_factor and pixelation_factor > 1:
        sh = max(1, H // pixelation_factor)
        sw = max(1, W // pixelation_factor)
        m = mask.float().unsqueeze(1)
        m = F.interpolate(m, size=(sh, sw), mode="area")
        m = F.interpolate(m, size=(H, W), mode="nearest")
        mask = (m.squeeze(1) > 127.5).to(torch.uint8) * 255
    if bernoulli_p and bernoulli_p > 0.0:
        flips = (torch.rand_like(mask, dtype=torch.float32) < bernoulli_p)
        mask = mask ^ (flips.to(torch.uint8) * 255)
    if erosion_px and erosion_px > 0:
        k = 2 * erosion_px + 1
        m = mask.float().unsqueeze(1)
        eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=erosion_px)
        mask = (eroded.squeeze(1) > 127.5).to(torch.uint8) * 255
    if motion_blur_px and motion_blur_px > 0.5:
        L = int(round(motion_blur_px))
        kx = torch.ones(1, 1, 1, 2 * L + 1, device=mask.device) / (2 * L + 1)
        m = mask.float().unsqueeze(1)
        m = F.conv2d(m, kx, padding=(0, L))
        mask = (m.squeeze(1) > 32.0).to(torch.uint8) * 255
    return mask


LAP_CONFIGS: List[Dict[str, Any]] = [
    {"name": "clean",              "pixelation_factor": 1, "bernoulli_p": 0.000, "erosion_px": 0, "motion_blur_px": 0.0},
    {"name": "lightly-pixelated",  "pixelation_factor": 2, "bernoulli_p": 0.005, "erosion_px": 0, "motion_blur_px": 0.0},
    {"name": "chunky+noisy",       "pixelation_factor": 4, "bernoulli_p": 0.020, "erosion_px": 1, "motion_blur_px": 0.0},
    {"name": "high-speed-blur",    "pixelation_factor": 2, "bernoulli_p": 0.010, "erosion_px": 0, "motion_blur_px": 8.0},
    {"name": "worst-case-onboard", "pixelation_factor": 4, "bernoulli_p": 0.040, "erosion_px": 1, "motion_blur_px": 6.0},
]


def _composite(
    mask_np: np.ndarray, out_size: tuple,
    lap_name: str, step_idx: int, drone_pos: np.ndarray,
    next_gate_idx: int, gates_passed: int, cfg_text: str,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    out_h, out_w = out_size
    rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    rgb[..., 2] = mask_np
    rgb[mask_np > 0, 0] = 32
    img = Image.fromarray(rgb).resize((out_w, out_h), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.text((24, 24),       f"LAP CONFIG: {lap_name}",
           fill=(255, 255, 255), font=font)
    d.text((24, 24 + 64),  f"step {step_idx:05d}  gate {next_gate_idx}/13  "
                            f"passed: {gates_passed}",
           fill=(180, 180, 180), font=font)
    d.text((24, 24 + 128), f"drone: x={drone_pos[0]:+.2f} "
                            f"y={drone_pos[1]:+.2f} z={drone_pos[2]:+.2f}",
           fill=(180, 180, 180), font=font)
    d.text((24, 24 + 192), cfg_text, fill=(120, 200, 255), font=font)
    return np.array(img)


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    video_dir = cfg.get("video_dir", "/tmp/segmask_pov")
    os.makedirs(video_dir, exist_ok=True)
    out_h, out_w = (2160, 3840) if cfg.get("video_4k", True) else (1080, 1920)
    fps = int(cfg.get("video_fps", 30))
    steps_per_lap = int(cfg.get("steps_per_lap", 600))
    laps_total = int(cfg.get("num_laps", len(LAP_CONFIGS)))

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Build the same transform stack training used so the checkpoint is fed
    # the obs shape it expects.
    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))
    action_transform = cfg.task.get("action_transform", None)
    if action_transform == "rate":
        transforms.append(RateController(base_env.controller, action_key=("agents", "action")))
    elif action_transform == "attitude":
        transforms.append(AttitudeController(base_env.controller, action_key=("agents", "action")))

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo, env.observation_spec, env.action_spec,
        env.reward_spec, device=base_env.device,
    )
    print(f"[play_pov_segmask] policy {cfg.algo.name} loaded; "
          f"checkpoint_path={getattr(cfg.algo, 'checkpoint_path', None)}")

    # Use the official GateSegmask sensor from staging EECS106B/gate-segmask:
    # pinhole projection + hollow-ring polygon fill (inner_ratio=0.6 makes the
    # gate look like a picture frame instead of a filled rectangle).
    from omni_drones.sensors import GateSegmask, GateSegmaskCfg
    seg_cfg = GateSegmaskCfg(
        resolution=(int(cfg.get("segmask_w", 128)),
                    int(cfg.get("segmask_h", 96))),
        fov_h_deg=float(cfg.get("segmask_fov_h_deg", 90.0)),
        gate_width=float(base_env.gate_width),
        gate_height=float(base_env.gate_height),
        inner_ratio=float(cfg.get("gate_inner_ratio", 0.6)),
        augment=False,
    )
    seg = GateSegmask(seg_cfg, device=base_env.device)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except Exception:
        font = ImageFont.load_default()

    print(f"[play_pov_segmask] env.num_envs={env.num_envs}, "
          f"max_ep_len={base_env.max_episode_length}, "
          f"laps_total={laps_total}, steps_per_lap={steps_per_lap}")

    td = env.reset()
    print(f"[play_pov_segmask] initial obs keys: {list(td.keys(True))[:10]}")

    lap_idx = 0
    lap_step = 0
    lap_frames: List[np.ndarray] = []
    cfg_lap = LAP_CONFIGS[0]
    cfg_text = (f"pixelation={cfg_lap['pixelation_factor']}  "
                f"bernoulli_p={cfg_lap['bernoulli_p']}  "
                f"erosion_px={cfg_lap['erosion_px']}  "
                f"motion_blur_px={cfg_lap['motion_blur_px']}")

    def _flush_lap(lap_idx: int, frames: List[np.ndarray], tag: str):
        if not frames:
            print(f"  no frames captured for lap {lap_idx + 1}; skipping write")
            return
        name = LAP_CONFIGS[lap_idx]["name"]
        out_path = os.path.join(video_dir, f"lap_{lap_idx + 1:02d}_{name}_{tag}.mp4")
        writer = imageio.get_writer(out_path, fps=fps, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  wrote {out_path}  ({len(frames)} frames @ {fps} fps)")

    total_steps = 0
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        while lap_idx < laps_total:
            td = policy(td)
            td = env.step(td)
            total_steps += 1
            lap_step += 1

            gate_w_pos, gate_w_rot = base_env.gates.get_world_poses()
            gate_e_pos, gate_e_rot = base_env.get_env_poses((gate_w_pos, gate_w_rot))
            drone_pos = base_env.drone.pos.squeeze(1)
            drone_rot = base_env.drone.rot.squeeze(1)

            # Hollow-ring binary segmask via pure pinhole projection.
            # Returns (N, H, W, 1) uint8.
            mask_4d = seg.render(drone_pos, drone_rot, gate_e_pos, gate_e_rot)
            mask = mask_4d.squeeze(-1)   # (N, H, W)

            # Apply per-lap perception degradations.
            mask = _degrade_mask(
                mask,
                pixelation_factor=cfg_lap["pixelation_factor"],
                bernoulli_p=cfg_lap["bernoulli_p"],
                erosion_px=cfg_lap["erosion_px"],
                motion_blur_px=cfg_lap["motion_blur_px"],
            )
            next_gate = int(base_env.gate_indices[0].item())
            gates_passed_this_lap = next_gate
            if base_env.track_completed[0].item():
                gates_passed_this_lap = base_env.num_gates

            frame = _composite(
                mask[0].cpu().numpy(),
                (out_h, out_w),
                cfg_lap["name"], total_steps,
                drone_pos[0].cpu().numpy(),
                next_gate, gates_passed_this_lap,
                cfg_text, font,
            )
            lap_frames.append(frame)

            if total_steps % 50 == 0:
                print(f"  step {total_steps}  lap_step {lap_step}/{steps_per_lap}  "
                      f"next_gate={next_gate}  pos={drone_pos[0].cpu().numpy()}")

            # Advance td to the next observation for the next iteration.
            td = td["next"]

            # Lap rollover: real lap or step budget.
            real_lap_done = bool(base_env.track_completed[0].item())
            chunk_done = lap_step >= steps_per_lap
            term = bool(td.get(("terminated",)).any().item()) if ("terminated",) in td.keys(True) else False
            trunc = bool(td.get(("truncated",)).any().item()) if ("truncated",) in td.keys(True) else False

            if real_lap_done or chunk_done or term or trunc:
                tag = ("completed" if real_lap_done else
                       ("terminated" if term else
                        ("truncated" if trunc else "chunked")))
                _flush_lap(lap_idx, lap_frames, tag)
                lap_idx += 1
                lap_step = 0
                lap_frames = []
                if lap_idx >= laps_total:
                    break
                cfg_lap = LAP_CONFIGS[lap_idx]
                cfg_text = (f"pixelation={cfg_lap['pixelation_factor']}  "
                            f"bernoulli_p={cfg_lap['bernoulli_p']}  "
                            f"erosion_px={cfg_lap['erosion_px']}  "
                            f"motion_blur_px={cfg_lap['motion_blur_px']}")
                print(f"\n[play_pov_segmask] starting lap {lap_idx + 1} "
                      f"({cfg_lap['name']})")
                td = env.reset()

    print(f"\n[play_pov_segmask] Done. {lap_idx} videos in {video_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
