# Copyright (c) 2025, C.K. Wolfe
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default PyTorch / Isaac Sim device: CUDA when available, else CPU."""

from __future__ import annotations

import torch


def default_sim_device() -> str:
    """Device string for Isaac Lab ``parse_env_cfg(..., device=...)`` and sim tensors.

    Returns ``cuda:0`` when a GPU is visible to PyTorch; otherwise ``cpu`` so
    headless unit tests still run without CUDA.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
