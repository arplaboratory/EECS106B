# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Download the heavy 5-inch racing drone assets from HuggingFace Hub.

The two largest binaries in the racing-drone bundle (the base_link COLLADA mesh
and the base configuration USD) live on HF Hub instead of in this git repo so
the repo stays sub-100MB-friendly without Git LFS. Run this script once after
cloning to pull them into ``omni_drones/robots/assets/drones/five_in/``.

Usage (from the repo root):
    python scripts/setup_racing_drone_assets.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ID = "ckwolfe/eecs106b-racing-drone-assets"
REPO_TYPE = "dataset"

ASSETS = [
    ("meshes/base_link.dae",
     "omni_drones/robots/assets/drones/five_in/meshes/base_link.dae"),
    ("configuration/5_in_drone_base.usd",
     "omni_drones/robots/assets/drones/five_in/configuration/5_in_drone_base.usd"),
]


def _eecs106b_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub not installed. Install with:\n"
            "    pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    root = _eecs106b_root()
    print(f"[setup] target root: {root}")
    print(f"[setup] HF repo:     {REPO_TYPE}/{REPO_ID}")

    for hf_path, repo_rel in ASSETS:
        dst = root / repo_rel
        if dst.exists() and dst.stat().st_size > 1024:
            print(f"[setup] skip (already present): {repo_rel} ({dst.stat().st_size/1e6:.1f} MB)")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[setup] fetching {hf_path} -> {repo_rel}")
        cached = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE, filename=hf_path,
        )
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(cached)
            print(f"[setup]   linked -> {cached}")
        except OSError:
            shutil.copyfile(cached, dst)
            print(f"[setup]   copied -> {dst}")

    print("[setup] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
