# ----------------------------------------------------------------------------
# Copyright (c) 2026 C. K. Wolfe. All rights reserved.
# NOT FREE TO USE.
# ----------------------------------------------------------------------------
"""Generate GateNet perception-training data from a trained PPO policy.

Per-step output (paper Section II-F SkyDreamer pipeline):
    out_dir/clean/step_NNNNNN.png      -- 64x64 uint8 binary clean mask
    out_dir/aug/step_NNNNNN.png        -- 64x64 uint8 mask after MaskErosion +
                                          RollingShutterWarp (training input)
    out_dir/meta/step_NNNNNN.json      -- drone/gate poses + camera rates
    out_dir/clean.mp4                  -- BW (NOT blue) review MP4 of clean masks
    out_dir/aug.mp4                    -- BW review MP4 of augmented masks

The mask renderer is `GateSegmask` (pinhole + hollow-ring fill, inner_ratio=0.6)
copied from staging EECS106B/gate-segmask. Augmentations are
`MaskErosion` and `RollingShutterWarp` from staging dreamt
(policy/gatenet/augmentations.py).

Native render resolution defaults to 64x64 -- the paper's nominal pinhole frame
(fx = fy = (25/64) * W). Crank `+segmask_h=N +segmask_w=N` higher if you need
more resolution for inspection; the policy never sees the mask anyway.

Invocation (omnidrones container):

    python scripts/gen_perception_data.py \\
        task=DroneRace algo=DroneRace \\
        algo.checkpoint_path=/workspace/omni_drones/scripts/wandb/goat-run-3-512-13.30/files/final_model.pt \\
        task.env.num_envs=1 \\
        task.env.max_episode_length=4000 \\
        headless=true \\
        +out_dir=/tmp/perception_data \\
        +num_steps=2000 \\
        +segmask_h=64 +segmask_w=64
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
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


def _save_mp4(frames: List[np.ndarray], path: str, fps: int = 30):
    if not frames:
        return
    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


def _bw_to_rgb(mask_2d_u8: np.ndarray, upscale_to: tuple | None = None) -> np.ndarray:
    """Convert a (H, W) uint8 {0, 255} mask to a (H, W, 3) RGB ndarray.
    Optionally NEAREST-resize to the target (H, W) for video output."""
    if upscale_to is not None:
        from PIL import Image
        img = Image.fromarray(mask_2d_u8).resize(
            (upscale_to[1], upscale_to[0]), Image.NEAREST,
        )
        mask_2d_u8 = np.array(img)
    return np.stack([mask_2d_u8] * 3, axis=-1)


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    out_dir = cfg.get("out_dir", "/tmp/perception_data")
    num_steps = int(cfg.get("num_steps", 2000))
    seg_h = int(cfg.get("segmask_h", 64))
    seg_w = int(cfg.get("segmask_w", 64))
    fov_h = float(cfg.get("segmask_fov_h_deg", 90.0))
    inner_ratio = float(cfg.get("gate_inner_ratio", 0.6))
    review_mp4_size = (
        int(cfg.get("review_h", 720)),
        int(cfg.get("review_w", 720)),
    )
    review_fps = int(cfg.get("review_fps", 30))
    save_per_step_meta = bool(cfg.get("save_meta", True))
    # Default: less-aggressive paper aug -- 1/4 the rolling-shutter warp range
    # and half-rate erosion. Override via CLI to revert to paper defaults.
    erosion_prob = float(cfg.get("erosion_prob", 0.25))
    rolling_shutter_s_hi = float(cfg.get("rolling_shutter_s", 0.005))
    only_save_best_lap = bool(cfg.get("only_save_best_lap", True))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "aug"), exist_ok=True)
    if save_per_step_meta:
        os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

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
    print(f"[gen] policy {cfg.algo.name} loaded; "
          f"checkpoint={getattr(cfg.algo, 'checkpoint_path', None)}")

    blur_sigma_lo = float(cfg.get("blur_sigma_lo", 0.3))
    blur_sigma_hi = float(cfg.get("blur_sigma_hi", 1.2))
    pixel_noise_std = float(cfg.get("pixel_noise_std", 12.0))

    from omni_drones.sensors import GateSegmask, GateSegmaskCfg
    seg_cfg = GateSegmaskCfg(
        resolution=(seg_w, seg_h),
        fov_h_deg=fov_h,
        gate_width=float(base_env.gate_width),
        gate_height=float(base_env.gate_height),
        inner_ratio=inner_ratio,
        # Gaussian blur built into GateSegmask is the paper's "blur sigma in
        # [lo, hi]" mask augmentation -- applied to the *augmented* path only.
        # We render twice: once clean, once with this turned on.
        augment=True,
        blur_sigma_lo=blur_sigma_lo,
        blur_sigma_hi=blur_sigma_hi,
        edge_shift_max_px=int(cfg.get("edge_shift_max_px", 0)),
    )
    seg_aug = GateSegmask(seg_cfg, device=base_env.device)
    seg_clean_cfg = GateSegmaskCfg(
        resolution=(seg_w, seg_h), fov_h_deg=fov_h,
        gate_width=float(base_env.gate_width),
        gate_height=float(base_env.gate_height),
        inner_ratio=inner_ratio, augment=False,
    )
    seg = GateSegmask(seg_clean_cfg, device=base_env.device)
    print(f"[gen] segmask res=({seg_h}, {seg_w}) fov_h={fov_h} "
          f"inner_ratio={inner_ratio}")

    # Paper Section II-F augmentations applied on top of the
    # already-Gaussian-blurred mask: MaskErosion + RollingShutterWarp +
    # additive Gaussian pixel noise.
    from omni_drones.sensors.gate_mask_augmentations import MaskErosion, RollingShutterWarp
    erode = MaskErosion(prob=erosion_prob, pool_size=2, hold_steps_avg=100).to(base_env.device)
    shutter = RollingShutterWarp(s_range=(0.0, rolling_shutter_s_hi)).to(base_env.device)

    def _gaussian_pixel_noise(mask_f: torch.Tensor, std_in_255: float) -> torch.Tensor:
        if std_in_255 <= 0.0:
            return mask_f
        noise = torch.randn_like(mask_f) * (std_in_255 / 255.0)
        return (mask_f + noise).clamp(0.0, 1.0)

    print(f"[gen] augs: GaussianBlur(sigma in [{blur_sigma_lo}, {blur_sigma_hi}]) + "
          f"MaskErosion(prob={erosion_prob}) + "
          f"RollingShutterWarp(s_max={rolling_shutter_s_hi}) + "
          f"PixelNoise(std={pixel_noise_std}/255)")
    print(f"[gen] num_steps={num_steps}; output -> {out_dir}")

    td = env.reset()
    # Per-step buffers for the CURRENT lap attempt; we flush them to disk
    # only after we know whether the lap was successful.
    cur_clean: List[np.ndarray] = []
    cur_aug: List[np.ndarray] = []
    cur_meta: List[dict] = []
    laps: List[dict] = []
    lap_start = 0
    max_gates = 0
    cur_start_gate = int(base_env.starting_gate_indices[0].item())

    def _flush_current_lap(end_step: int, term: bool, trunc: bool, completed: bool):
        nonlocal max_gates
        gates = max_gates
        info = dict(
            start=lap_start, end=end_step, length=end_step - lap_start,
            gates_passed=gates, completed=completed,
            start_gate=cur_start_gate,
            terminated=term, truncated=trunc,
            # Score: prefer real laps from gate 0, then most gates, then length.
            score=(int(completed and cur_start_gate == 0) * 1_000_000
                   + int(completed) * 10_000
                   + gates * 100
                   + (end_step - lap_start)),
        )
        info["frames_clean"] = list(cur_clean)
        info["frames_aug"] = list(cur_aug)
        info["meta"] = list(cur_meta)
        laps.append(info)
        print(f"[gen]   lap done: start={lap_start} end={end_step} "
              f"start_gate={cur_start_gate} gates={gates} "
              f"completed={completed} term={term} trunc={trunc}")

    # Match training-time exploration: the policy was trained with the
    # stochastic actor, so deterministic playback (MODE) underperforms here.
    # Keep RANDOM unless the user explicitly asks for deterministic.
    explo_mode_str = str(cfg.get("explo", "random")).lower()
    explo_map = {
        "random": ExplorationType.RANDOM,
        "mode": ExplorationType.MODE,
        "mean": ExplorationType.MEAN,
    }
    exploration = explo_map.get(explo_mode_str, ExplorationType.RANDOM)
    print(f"[gen] exploration mode = {explo_mode_str}")

    with set_exploration_type(exploration), torch.no_grad():
        for step_i in range(num_steps):
            td = policy(td)
            td = env.step(td)

            gate_w_pos, gate_w_rot = base_env.gates.get_world_poses()
            gate_e_pos, gate_e_rot = base_env.get_env_poses((gate_w_pos, gate_w_rot))
            drone_pos = base_env.drone.pos.squeeze(1)
            drone_rot = base_env.drone.rot.squeeze(1)

            # Clean (no blur, no noise) for ground truth.
            mask_4d = seg.render(drone_pos, drone_rot, gate_e_pos, gate_e_rot)
            clean_mask = mask_4d.squeeze(-1)[0]   # (H, W) uint8
            # Augmented base: GateSegmask's Gaussian blur (sigma random per call)
            # + edge shift (paper-style soft-mask augmentation).
            mask_aug_4d = seg_aug.render(drone_pos, drone_rot, gate_e_pos, gate_e_rot)
            aug_base = mask_aug_4d.squeeze(-1)[0]  # (H, W) uint8

            # Camera-frame yaw and pitch rates for the rolling-shutter warp.
            try:
                ang_vel_body = base_env.drone.vel[..., 0, 3:6][0]  # (3,)
            except Exception:
                ang_vel_body = torch.zeros(3, device=base_env.device)
            # body angular velocity components -> camera-frame rates.
            # Camera convention: drone +x forward maps to camera +z forward,
            # so yaw rate around body z = pitch rate around camera y, and
            # pitch rate around body y = -yaw rate around camera x.
            yaw_rate_cam = ang_vel_body[2]      # body wz
            pitch_rate_cam = ang_vel_body[1]    # body wy

            # Apply paper augmentations on top of the already-blurred soft mask.
            mask_f = aug_base.float().unsqueeze(0).unsqueeze(0) / 255.0
            sample = {"mask": mask_f, "yaw_rate_cam": yaw_rate_cam.unsqueeze(0),
                      "pitch_rate_cam": pitch_rate_cam.unsqueeze(0)}
            sample = erode(sample)
            sample = shutter(sample)
            aug_mask_f = sample["mask"].clamp(0.0, 1.0)
            # Additive Gaussian pixel noise (the missing "gaussian" channel).
            aug_mask_f = _gaussian_pixel_noise(aug_mask_f, pixel_noise_std)
            aug_mask = (aug_mask_f.squeeze(0).squeeze(0) * 255.0).to(torch.uint8)

            clean_np = clean_mask.cpu().numpy()
            aug_np = aug_mask.cpu().numpy()

            next_gate = int(base_env.gate_indices[0].item())
            track_done = bool(base_env.track_completed[0].item())
            # gates_passed = (next_gate - start_gate) clipped, +1 if track done.
            gp = max(0, next_gate - cur_start_gate) + (1 if track_done else 0)
            max_gates = max(max_gates, gp)

            cur_clean.append(_bw_to_rgb(clean_np, upscale_to=review_mp4_size))
            cur_aug.append(_bw_to_rgb(aug_np, upscale_to=review_mp4_size))
            cur_meta.append({
                "step": step_i,
                "drone_pos_world": drone_pos[0].cpu().tolist(),
                "drone_rot_wxyz": drone_rot[0].cpu().tolist(),
                "ang_vel_body_xyz": ang_vel_body.cpu().tolist(),
                "next_gate_idx": next_gate,
                "track_completed": track_done,
                "clean_png": clean_np.tolist() if clean_np.size <= 4096 else None,
            })

            if step_i % 100 == 0:
                px = int(clean_mask.sum().item() // 255)
                print(f"  step {step_i:5d}  next_gate={next_gate} "
                      f"  drone={drone_pos[0].cpu().numpy().round(2).tolist()} "
                      f"  mask_pix={px}  track_completed={track_done}")

            td = td["next"]
            term = bool(td.get(("terminated",)).any().item()) if ("terminated",) in td.keys(True) else False
            trunc = bool(td.get(("truncated",)).any().item()) if ("truncated",) in td.keys(True) else False
            if term or trunc:
                _flush_current_lap(step_i, term, trunc, track_done)
                cur_clean = []; cur_aug = []; cur_meta = []
                lap_start = step_i + 1
                max_gates = 0
                td = env.reset()
                cur_start_gate = int(base_env.starting_gate_indices[0].item())

    # If the run hits num_steps without an episode boundary, flush the
    # in-progress lap so it's still considered.
    if cur_clean:
        _flush_current_lap(num_steps - 1, False, True, False)

    # 4) Pick the BEST lap. Score prioritises real completions, then gates
    # passed, then length.  If no lap completed, save the longest run.
    print(f"[gen] {len(laps)} candidate lap(s):")
    for i, L in enumerate(laps):
        print(f"  [{i}] start={L['start']:5d} end={L['end']:5d} len={L['length']:4d} "
              f"gates={L['gates_passed']:3d} completed={L['completed']} score={L['score']}")
    if not laps:
        print("[gen] no laps recorded; nothing to save.")
        simulation_app.close()
        return

    # Multi-lap mode (default): keep all complete laps that started at gate 0
    # and stitch them into a single MP4, with a 1-second fade between them
    # so playback shows lap N -> reset -> lap N+1.  This is what the user
    # asked for ("do multiple laps").
    multilap = bool(cfg.get("multilap", True))
    if multilap:
        good = [L for L in laps if L["completed"] and L.get("start_gate", 0) == 0
                and len(L["frames_clean"]) > 100]
        if not good:
            # Fall back to all completed laps regardless of spawn gate.
            good = [L for L in laps if L["completed"] and len(L["frames_clean"]) > 100]
        if not good:
            print("[gen] no completed laps; falling back to best single attempt.")
            best = max(laps, key=lambda L: L["score"])
            good = [best]
        clean_all: List[np.ndarray] = []
        aug_all: List[np.ndarray] = []
        # Insert a black gap frame between concatenated laps so the cut is obvious.
        h, w = good[0]["frames_clean"][0].shape[:2]
        gap_clean = [np.zeros((h, w, 3), dtype=np.uint8)] * review_fps
        for i, L in enumerate(good):
            print(f"[gen]   stitching lap {i + 1}/{len(good)}: "
                  f"start={L['start']} len={L['length']} gates={L['gates_passed']}")
            clean_all.extend(L["frames_clean"])
            aug_all.extend(L["frames_aug"])
            if i + 1 < len(good):
                clean_all.extend(gap_clean)
                aug_all.extend(gap_clean)
        print(f"[gen] stitched {len(good)} lap(s) -> {len(clean_all)} frames")
        _save_mp4(clean_all, os.path.join(out_dir, "clean.mp4"), review_fps)
        _save_mp4(aug_all, os.path.join(out_dir, "aug.mp4"), review_fps)
        best = good[0]   # used by metadata below
    else:
        best = max(laps, key=lambda L: L["score"])
        print(f"[gen] best lap: start={best['start']} end={best['end']} "
              f"len={best['length']} gates={best['gates_passed']} "
              f"completed={best['completed']}")
        _save_mp4(best["frames_clean"], os.path.join(out_dir, "clean.mp4"), review_fps)
        _save_mp4(best["frames_aug"], os.path.join(out_dir, "aug.mp4"), review_fps)

    # Save per-step PNGs (training data) for the best lap only.
    for i, m in enumerate(best["meta"]):
        clean_arr = np.array(m["clean_png"], dtype=np.uint8) if m["clean_png"] is not None else None
        if clean_arr is None:
            # Reconstruct from review frame by extracting top-left native sample.
            # (Native PNGs are saved separately below from the in-memory aug ndarrays.)
            continue
        imageio.imwrite(
            os.path.join(out_dir, "clean", f"step_{i:06d}.png"), clean_arr,
        )
        if save_per_step_meta:
            with open(os.path.join(out_dir, "meta", f"step_{i:06d}.json"), "w") as f:
                json.dump({k: v for k, v in m.items() if k != "clean_png"}, f)

    # 5) Top-level dataset metadata.
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump({
            "num_steps_collected": num_steps,
            "segmask_h": seg_h,
            "segmask_w": seg_w,
            "fov_h_deg": fov_h,
            "inner_ratio": inner_ratio,
            "erosion_prob": erosion_prob,
            "rolling_shutter_s": rolling_shutter_s_hi,
            "review_size": review_mp4_size,
            "review_fps": review_fps,
            "checkpoint_path": str(getattr(cfg.algo, "checkpoint_path", None)),
            "task": cfg.task.name,
            "laps_attempted": len(laps),
            "best_lap": {
                "start": best["start"], "end": best["end"],
                "length": best["length"],
                "gates_passed": best["gates_passed"],
                "completed": best["completed"],
            },
        }, f, indent=2)
    print(f"[gen] Done. Output dir: {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
