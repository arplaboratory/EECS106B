"""Section II-F GateNet augmentations.

Paper pipeline: shot noise (sigma=40 for 1 ms exposure), mask erosion (50%,
~100-env-step hold), rolling-shutter affine warp parametrized by the camera's
yaw and pitch rate.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


try:  # TODO: prefer kornia if available.
    from kornia.geometry.transform import warp_affine as _kornia_warp_affine  # type: ignore
    _HAS_KORNIA = True
except Exception:  # pragma: no cover
    _kornia_warp_affine = None
    _HAS_KORNIA = False


class ShotNoise(nn.Module):
    """Additive Gaussian shot noise in 0-255 pixel range.

    Paper uses sigma=40 (stronger than the usual sigma=25) to match the very
    short 1 ms exposure used onboard.
    """

    def __init__(self, std: float = 40.0) -> None:
        super().__init__()
        self.std = float(std)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "rgb" in sample:
            rgb = sample["rgb"]
            noise = torch.randn_like(rgb) * self.std
            sample = dict(sample)
            sample["rgb"] = (rgb + noise).clamp(0.0, 255.0)
        return sample


class MaskErosion(nn.Module):
    """50% chance 1-pixel erosion via average-pool + threshold.

    The erosion state is stable across ~100 env steps (paper): a running hold
    counter decides when to resample.
    """

    def __init__(self, prob: float = 0.5, pool_size: int = 2, hold_steps_avg: int = 100) -> None:
        super().__init__()
        self.prob = float(prob)
        self.pool_size = int(pool_size)
        self.hold_steps_avg = int(hold_steps_avg)
        # Stateful: decremented each forward; on zero we resample.
        self.register_buffer("_hold", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_active", torch.zeros(1, dtype=torch.bool), persistent=False)

    def _maybe_resample(self) -> None:
        if int(self._hold.item()) <= 0:
            active = torch.rand(1).item() < self.prob
            self._active.fill_(active)
            # Geometric-ish dwell: uniform in [0.5x, 1.5x] average steps.
            low = max(1, int(0.5 * self.hold_steps_avg))
            high = max(low + 1, int(1.5 * self.hold_steps_avg))
            self._hold.fill_(torch.randint(low, high, (1,)).item())
        self._hold -= 1

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "mask" not in sample:
            return sample
        self._maybe_resample()
        if not bool(self._active.item()):
            return sample
        mask = sample["mask"]
        if mask.dim() == 3:
            mask_b = mask.unsqueeze(0)
        else:
            mask_b = mask
        # Pad by (k-1)//2 on one side and k//2 on the other to keep exact size
        # for arbitrary k, then stride-1 avg_pool without any extra padding.
        pad_lo = (self.pool_size - 1) // 2
        pad_hi = self.pool_size // 2
        padded = F.pad(mask_b.float(), (pad_lo, pad_hi, pad_lo, pad_hi), mode="replicate")
        pooled = F.avg_pool2d(padded, kernel_size=self.pool_size, stride=1, padding=0)
        # Erosion: keep only fully-inside regions.
        eroded = (pooled >= 0.999).to(mask.dtype)
        if mask.dim() == 3:
            eroded = eroded.squeeze(0)
        sample = dict(sample)
        sample["mask"] = eroded
        return sample


class RollingShutterWarp(nn.Module):
    """Affine rolling-shutter warp (paper Section II-F).

    The warp matrix is
        A = [[1,     -s * r_c, (W/2) * s * r_c],
             [0,  1 + s * q_c, -(H/2) * s * q_c]]
    where s ~ U(s_range) is the shutter-delay coefficient (seconds), and
    r_c, q_c are yaw- and pitch-rate in camera frame (rad/s).
    """

    def __init__(self, s_range=(0.0, 0.02), rate_source: str = "env") -> None:
        super().__init__()
        self.s_low, self.s_high = float(s_range[0]), float(s_range[1])
        self.rate_source = rate_source

    def _sample_s(self, device: torch.device) -> torch.Tensor:
        u = torch.rand(1, device=device)
        return self.s_low + (self.s_high - self.s_low) * u

    @staticmethod
    def _affine_matrix(s: torch.Tensor, r_c: torch.Tensor, q_c: torch.Tensor,
                       W: int, H: int) -> torch.Tensor:
        """Return 2x3 affine in pixel coords."""
        m11 = torch.ones_like(s)
        m12 = -s * r_c
        m13 = (W / 2.0) * s * r_c
        m21 = torch.zeros_like(s)
        m22 = 1.0 + s * q_c
        m23 = -(H / 2.0) * s * q_c
        row0 = torch.stack([m11, m12, m13], dim=-1)
        row1 = torch.stack([m21, m22, m23], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (..., 2, 3)

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "mask" not in sample:
            return sample
        mask = sample["mask"]
        if mask.dim() == 3:
            mask_b = mask.unsqueeze(0)  # (1, C, H, W)
        elif mask.dim() == 4:
            mask_b = mask
        else:
            return sample
        N, C, H, W = mask_b.shape
        device = mask_b.device

        r_c = sample.get("yaw_rate_cam", torch.zeros(N, device=device))
        q_c = sample.get("pitch_rate_cam", torch.zeros(N, device=device))
        r_c = r_c.to(device).reshape(-1).float()
        q_c = q_c.to(device).reshape(-1).float()
        if r_c.numel() == 1 and N > 1:
            r_c = r_c.expand(N)
        if q_c.numel() == 1 and N > 1:
            q_c = q_c.expand(N)
        s = self._sample_s(device).expand(N)

        A = self._affine_matrix(s, r_c, q_c, W, H)  # (N, 2, 3)

        if _HAS_KORNIA:
            warped = _kornia_warp_affine(mask_b.float(), A, dsize=(H, W))
        else:
            # Normalize pixel-coord affine to normalized coords for grid_sample.
            # grid_sample expects theta in normalized [-1, 1] frame.
            theta = _pixel_affine_to_normalized(A, H, W)
            grid = F.affine_grid(theta, size=mask_b.shape, align_corners=False)
            warped = F.grid_sample(mask_b.float(), grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=False)

        if mask.dim() == 3:
            warped = warped.squeeze(0)
        sample = dict(sample)
        sample["mask"] = warped.to(mask.dtype)
        return sample


def _pixel_affine_to_normalized(A: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Convert a pixel-space 2x3 affine A (y = A x) to a normalized-coord theta
    suitable for torch.nn.functional.affine_grid."""
    N = A.shape[0]
    # Normalized->pixel: p = N_to_P * n  where N_to_P = diag(W/2, H/2) and +cx
    # with cx=W/2-0.5 style. Use align_corners=False mapping.
    sx = W / 2.0
    sy = H / 2.0
    # theta maps output-normalized coord to input-normalized coord.
    # A operates in pixel coords: p_in = A * [p_out; 1].
    # Let P = diag(sx, sy), b = [sx-0.5, sy-0.5]. Then
    #   p_out_pix = P * n_out + b  (approx., align_corners=False)
    # n_in = P^{-1} (A_rot * (P n_out + b) + A_t - b)
    #      = P^{-1} A_rot P n_out + P^{-1} (A_rot b + A_t - b)
    A_rot = A[:, :, :2]
    A_t = A[:, :, 2]
    P = torch.tensor([[sx, 0.0], [0.0, sy]], device=A.device, dtype=A.dtype)
    P_inv = torch.tensor([[1.0 / sx, 0.0], [0.0, 1.0 / sy]], device=A.device, dtype=A.dtype)
    b = torch.tensor([sx - 0.5, sy - 0.5], device=A.device, dtype=A.dtype)
    theta_rot = P_inv @ A_rot @ P  # (N, 2, 2)
    theta_t = (P_inv @ (A_rot @ b.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)).squeeze(-1) \
              + (P_inv @ A_t.unsqueeze(-1)).squeeze(-1) \
              - (P_inv @ b.unsqueeze(-1)).squeeze(-1)
    theta = torch.cat([theta_rot, theta_t.unsqueeze(-1)], dim=-1)
    return theta


class _PaperPipeline(nn.Module):
    """Sequential paper pipeline: rolling shutter -> mask erosion -> shot noise."""

    def __init__(self, W: int, H: int) -> None:
        super().__init__()
        self.W = int(W)
        self.H = int(H)
        self.rolling_shutter = RollingShutterWarp()
        self.erosion = MaskErosion()
        self.shot_noise = ShotNoise()

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = self.rolling_shutter(sample)
        sample = self.erosion(sample)
        sample = self.shot_noise(sample)
        return sample


def build_paper_augmentation_pipeline(W: int, H: int) -> nn.Module:
    """Factory: paper-exact augmentation pipeline for (W, H) images/masks."""
    return _PaperPipeline(W, H)
