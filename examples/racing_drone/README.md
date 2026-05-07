# `racing_drone/` — examples for the 5-inch racing drone

Demos that use the IsaacLab `FIVE_IN_DRONE` articulation cfg and (optionally)
compare it side-by-side with `iris.usd`.

| File | What it does |
|------|--------------|
| `01_play.py` | Minimal: spawn `FIVE_IN_DRONE`, freefall under gravity, exercise the `omni_drones.robots.dynamics` primitives (Allocation/Motor/AttitudeController/BodyRateController). |
| `02_hover.py` | Hold a fixed altitude with thrust + body-frame torque. Saves a trajectory plot. |
| `03_fig8_iris.py` | Iris flying a horizontal Lissajous fig-8 as a kinematic puppet, captured by a fixed external camera. |
| `04_fig8_compare.py` | Iris + 5-inch quad following the same fig-8 reference; each tracker is rate-limited by its own physical envelope so the agility difference shows up naturally. 4K render by default. |
| `05_fig8_aggressive.py` | Same as `04` but with a tighter, faster fig-8 (radius 4 m, period 2.5 s) and a low chase camera. |

## Heavy assets are on Hugging Face

The 5-inch drone bundle's two largest binaries
(`meshes/base_link.dae` ≈ 155 MB and `configuration/5_in_drone_base.usd`
≈ 103 MB) are NOT in this git repo. They live in the public dataset
**[`ckwolfe/eecs106b-racing-drone-assets`](https://huggingface.co/datasets/ckwolfe/eecs106b-racing-drone-assets)**.

Run once after cloning to fetch them:

```bash
python scripts/setup_racing_drone_assets.py
```

The fetcher reads `HF_TOKEN` from your environment if the dataset is private
in the future; the current dataset is public, so anonymous access works too.

## Environment

Copy `.env.example` to `.env` and fill in your tokens:

```bash
cp .env.example .env
# then edit .env
```

## Running

All commands assume you are at the repo root and have entered the
`omnidrones` distrobox container (see `SETUP_distrobox.md`).

```bash
# minimum: spawn + dynamics smoke test
python examples/racing_drone/01_play.py --headless

# hover the racing drone
python examples/racing_drone/02_hover.py --headless --steps=600

# 4K side-by-side comparison (8 s @ 60 fps; pass --frames to extend)
python examples/racing_drone/04_fig8_compare.py --headless --enable_cameras
```
