# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Synthetic gate segmentation mask sensor.

Renders a binary mask of racing-gate openings via a pure pinhole projection +
convex-polygon fill on the GPU. No Replicator / RTX rendering pass is required,
so the sensor is cheap enough to run alongside thousands of parallel Isaac envs
and is deterministic w.r.t. drone / gate poses (no per-step renderer noise).

Adapted from the SkyDreamer-style synthetic mask used in the staging
``omnidrone-106b`` repo (``drone_race_skydreamer.py``).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse


@dataclass
class GateSegmaskCfg:
    """Configuration for ``GateSegmask``.

    Attributes:
        resolution: (W, H) of the rendered mask.
        fov_h_deg: horizontal field of view of the virtual pinhole camera.
        gate_width: width of the gate opening (metres).
        gate_height: height of the gate opening (metres). Used to build the
            four corners of the opening (bottom-centre origin convention).
        inner_ratio: 0.0 -> filled quad, 0.95 -> very thin ring. The mask is a
            hollow ring obtained by subtracting an inner scaled quad.
        augment: if True, apply random Gaussian blur and sub-pixel edge jitter
            (see ``blur_sigma_lo/hi`` and ``edge_shift_max_px``).
        blur_sigma_lo / blur_sigma_hi: range of Gaussian blur sigma sampled per
            call when ``augment`` is True.
        edge_shift_max_px: max integer pixel-roll applied per axis when
            ``augment`` is True.
    """

    resolution: tuple = (64, 48)
    fov_h_deg: float = 90.0
    gate_width: float = 1.0
    gate_height: float = 2.0
    inner_ratio: float = 0.6
    augment: bool = False
    blur_sigma_lo: float = 0.3
    blur_sigma_hi: float = 2.0
    edge_shift_max_px: int = 0


class GateSegmask:
    """Synthetic binary segmentation-mask renderer for racing gates.

    Usage mirrors :class:`omni_drones.sensors.camera.Camera`: construct from a
    config, then call ``render(...)`` each step with current drone & gate
    poses. There is no ``spawn`` / ``initialize`` step because no USD prim is
    created -- the mask is a pure GPU computation.

    Example:
        >>> sensor = GateSegmask(GateSegmaskCfg(resolution=(64, 48)), device="cuda")
        >>> mask = sensor.render(drone_pos, drone_rot, gate_pos, gate_rot)
        >>> # mask: (N, H, W, 1) uint8, 255 = gate pixel, 0 = background
    """

    def __init__(self, cfg: GateSegmaskCfg, device):
        self.cfg = cfg
        self.device = device
        self.W, self.H = int(cfg.resolution[0]), int(cfg.resolution[1])
        self.inner_ratio = float(max(0.0, min(0.95, cfg.inner_ratio)))

        fov_rad = cfg.fov_h_deg * math.pi / 180.0
        self.cam_fx = self.W / (2.0 * math.tan(fov_rad / 2.0))
        self.cam_fy = self.cam_fx
        self.cam_cx = self.W / 2.0
        self.cam_cy = self.H / 2.0

        # Gate opening corners in gate-local frame (bottom-centre origin):
        # gate plane is the local YZ plane; +x is the through-direction.
        W_g, H_g = float(cfg.gate_width), float(cfg.gate_height)
        self.gate_corners_local = torch.tensor(
            [
                [0.0, -W_g / 2, 0.0],
                [0.0,  W_g / 2, 0.0],
                [0.0,  W_g / 2, H_g],
                [0.0, -W_g / 2, H_g],
            ],
            device=device,
        )  # (4, 3)

        # Pixel grid (u, v) flattened to (H*W, 2)
        v_grid, u_grid = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=device),
            torch.arange(self.W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        self._pixel_coords = torch.stack(
            [u_grid.reshape(-1), v_grid.reshape(-1)], dim=-1
        )  # (H*W, 2)

    @property
    def shape(self):
        """(H, W) -- matches :class:`Camera.shape` convention."""
        return (self.H, self.W)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _gate_corners_in_body_frame(self, gate_pos, gate_rot, drone_pos, drone_rot):
        """Compute 4 gate-opening corners in drone body frame.

        Args:
            gate_pos: (N, 3) gate origin in world/env frame.
            gate_rot: (N, 4) gate orientation quaternion.
            drone_pos: (N, 3) drone position.
            drone_rot: (N, 4) drone orientation quaternion.

        Returns:
            (N, 4, 3) corners in body frame.
        """
        N = gate_pos.shape[0]
        corners = self.gate_corners_local  # (4, 3)
        q_g = gate_rot.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        c_e = corners.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)
        cw = quat_rotate(q_g, c_e).reshape(N, 4, 3) + gate_pos.unsqueeze(1)
        rel = cw - drone_pos.unsqueeze(1)
        q_d = drone_rot.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        return quat_rotate_inverse(q_d, rel.reshape(-1, 3)).reshape(N, 4, 3)

    # ------------------------------------------------------------------
    # Mask augmentation (optional)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 1e-6:
            return x
        radius = max(1, int(math.ceil(3.0 * sigma)))
        k = 2 * radius + 1
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        g = g / g.sum()
        kernel = (g[:, None] @ g[None, :]).view(1, 1, k, k)
        return F.conv2d(x, kernel, padding=radius)

    def _augment(self, mask_u8: torch.Tensor) -> torch.Tensor:
        """Blur + optional sub-pixel edge jitter on a (N, H, W, 1) uint8 mask."""
        if not self.cfg.augment:
            return mask_u8
        x = mask_u8.to(dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, 1, H, W)
        if self.cfg.blur_sigma_hi > 1e-6:
            sigma = self.cfg.blur_sigma_lo + (
                self.cfg.blur_sigma_hi - self.cfg.blur_sigma_lo
            ) * torch.rand((), device=x.device, dtype=x.dtype)
            x = self._gaussian_blur(x, float(sigma.item()))
        s = self.cfg.edge_shift_max_px
        if s > 0:
            dx = int(torch.randint(-s, s + 1, (1,), device=x.device).item())
            dy = int(torch.randint(-s, s + 1, (1,), device=x.device).item())
            if dx != 0 or dy != 0:
                x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        x = x.clamp(0.0, 1.0)
        return (x.permute(0, 2, 3, 1) * 255.0).to(torch.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        drone_pos: torch.Tensor,
        drone_rot: torch.Tensor,
        gate_pos: torch.Tensor,
        gate_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Render binary gate segmentation masks.

        Args:
            drone_pos: (N, 3) drone positions.
            drone_rot: (N, 4) drone orientation quaternions (w, x, y, z).
            gate_pos: (N, G, 3) gate origin positions (bottom-centre).
            gate_rot: (N, G, 4) gate orientation quaternions.

        Returns:
            (N, H, W, 1) uint8 tensor. 255 = gate-ring pixel, 0 = background.
        """
        if drone_pos.dim() != 2 or drone_pos.shape[-1] != 3:
            raise ValueError(f"drone_pos must be (N, 3); got {tuple(drone_pos.shape)}")
        if drone_rot.dim() != 2 or drone_rot.shape[-1] != 4:
            raise ValueError(f"drone_rot must be (N, 4); got {tuple(drone_rot.shape)}")
        if gate_pos.dim() != 3 or gate_pos.shape[-1] != 3:
            raise ValueError(f"gate_pos must be (N, G, 3); got {tuple(gate_pos.shape)}")
        if gate_rot.dim() != 3 or gate_rot.shape[-1] != 4:
            raise ValueError(f"gate_rot must be (N, G, 4); got {tuple(gate_rot.shape)}")

        N = drone_pos.shape[0]
        G = gate_pos.shape[1]
        H, W = self.H, self.W

        mask = torch.zeros(N, H, W, dtype=torch.uint8, device=self.device)
        pixels = self._pixel_coords  # (H*W, 2)

        for g in range(G):
            cb = self._gate_corners_in_body_frame(
                gate_pos[:, g], gate_rot[:, g], drone_pos, drone_rot
            )  # (N, 4, 3) in body frame
            # body -> camera convention (drone +x forward, +y left, +z up):
            #   cam_x = -body_y    (camera +x right)
            #   cam_y = -body_z    (camera +y down)
            #   cam_z =  body_x    (camera +z forward)
            cam_z = cb[..., 0]
            in_front = (cam_z > 0.1).all(dim=1)        # (N,)
            z_safe = cam_z.clamp(min=0.1)
            u_proj = self.cam_fx * (-cb[..., 1]) / z_safe + self.cam_cx
            v_proj = self.cam_fy * (-cb[..., 2]) / z_safe + self.cam_cy
            corners_2d = torch.stack([u_proj, v_proj], dim=-1)  # (N, 4, 2)

            # Convex polygon fill via consistent cross-product sign.
            e_start = corners_2d
            e_end = torch.roll(corners_2d, -1, dims=1)
            d = e_end - e_start                                # (N, 4, 2)
            p = pixels.unsqueeze(0).unsqueeze(0)               # (1, 1, H*W, 2)
            pa = p - e_start.unsqueeze(2)                      # (N, 4, H*W, 2)
            d_exp = d.unsqueeze(2)                             # (N, 4, 1, 2)
            cross = d_exp[..., 0] * pa[..., 1] - d_exp[..., 1] * pa[..., 0]
            inside_outer = (cross >= 0).all(dim=1) | (cross <= 0).all(dim=1)

            # Hollow ring: subtract a scaled-down inner quad.
            if self.inner_ratio > 0.0:
                center_2d = corners_2d.mean(dim=1, keepdim=True)
                inner = center_2d + self.inner_ratio * (corners_2d - center_2d)
                i_start = inner
                i_end = torch.roll(inner, -1, dims=1)
                di = i_end - i_start
                pai = p - i_start.unsqueeze(2)
                di_exp = di.unsqueeze(2)
                cross_i = (
                    di_exp[..., 0] * pai[..., 1] - di_exp[..., 1] * pai[..., 0]
                )
                inside_inner = (cross_i >= 0).all(dim=1) | (cross_i <= 0).all(dim=1)
                ring = inside_outer & (~inside_inner)
            else:
                ring = inside_outer

            gate_mask = ring.reshape(N, H, W) & in_front.unsqueeze(-1).unsqueeze(-1)
            mask = mask | (gate_mask.to(torch.uint8) * 255)

        mask = mask.unsqueeze(-1)  # (N, H, W, 1)
        return self._augment(mask)
