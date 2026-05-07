# Racing drone (5-inch)

Higher-fidelity quadrotor model for racing-style tasks, ported from the lab's
private `isaac_drone_racer` work into this public class repo.

## What's in the PR

| Path | What it is |
|------|-----------|
| `omni_drones/robots/assets/drones/five_in/` | On-disk bundle: `5_in_drone.usd`, `urdf/5_in_drone.urdf`, `meshes/{base_link,prop}.dae`, `configuration/{5_in_drone_base,_physics,_robot,_sensor}.usd`. URDF was re-authored with link names `base_link` + `rotor_[0-3]` and joint names `rotor_[0-3]_joint`; USD was regenerated via `isaacsim.asset.importer.urdf`. |
| `omni_drones/robots/assets/five_in_drone.py` | IsaacLab `ArticulationCfg` (`FIVE_IN_DRONE`) for the 5-inch quad. Loads the bundle's USD and sets sim/articulation properties. |
| `omni_drones/robots/dynamics/` | `Allocation`, `Motor`, `AttitudeController`, `BodyRateController` (PyTorch, batched over `num_envs`). Self-contained — pulls `default_sim_device` from a sibling `_device.py`. |
| `examples/racing_drone/` | All 5-inch demos (`01_play.py`, `02_hover.py`, `03_fig8_iris.py`, `04_fig8_compare.py`, `05_fig8_aggressive.py`) + a folder README. |
| `scripts/setup_racing_drone_assets.py` | Run-once fetcher: pulls the heavy binaries from the public HF dataset `ckwolfe/eecs106b-racing-drone-assets` into the bundle dir. |
| `.env.example` | Template for `WANDB_API_KEY` / `HF_TOKEN`; copy to `.env` and fill in. |

## Use it

```python
from omni_drones.robots.assets.five_in_drone import FIVE_IN_DRONE
robot_cfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

```python
from omni_drones.robots.dynamics import (
    Allocation, AttitudeController, BodyRateController, Motor,
)
```

Run the demos (inside the distrobox container, from the repo root):

```bash
# fetch heavy USD/DAE assets (one-time)
python scripts/setup_racing_drone_assets.py

# minimum sanity demo
python examples/racing_drone/01_play.py --headless --steps=200

# 4K side-by-side comparison
python examples/racing_drone/04_fig8_compare.py --headless --enable_cameras
```

## Drone parameters (URDF + tuning)

* Mass: 0.5 kg
* Inertia (diag): `ixx=0.003, iyy=0.003, izz=0.006`
* Arm length (COM -> motor hub, planar): 0.1249 m (X-config; URDF joint XY = +/-0.0883)
* Rotor angles: `[pi/4, -pi/4, 3pi/4, -3pi/4]`
* Rotor directions: `[+1, -1, +1, -1]`
* Force constant: 4.8e-7; moment constant: 2.0e-9
* Max rotor speed: 5000 rad/s

## Asset notes

`meshes/base_link.dae` (~155 MB) and `configuration/5_in_drone_base.usd` (~103 MB)
are large binary assets and are NOT committed to this git repo. They live in the
public HF dataset
[`ckwolfe/eecs106b-racing-drone-assets`](https://huggingface.co/datasets/ckwolfe/eecs106b-racing-drone-assets)
and are pulled into the bundle directory by
`scripts/setup_racing_drone_assets.py`.

## Known follow-ups (not in this PR)

* **`omni_drones.MultirotorBase` adapter.** A subclass + YAML config to register
  `FiveIn` in the existing drone registry alongside `Crazyflie`/`ArplRace`/etc
  was scoped but deferred. The blocker is a structural mismatch between the
  IsaacSim URDF importer's output (an extra `/base_link/` wrapper Xform) and
  the iris-style hierarchy that `MultirotorBase` expects (rigid bodies as direct
  children of the spawn root). Fixing this needs either a USD post-processing
  pass to flatten the hierarchy, or a `FiveIn` subclass that overrides
  `prim_paths_expr` and the rotor-joint discovery logic in
  `multirotor.py:initialize`. Once landed, `FiveIn` works with `scripts/play.py`,
  `scripts/train.py`, and `LeePositionController`.
* **PPO training task.** The private `isaac_drone_racer` repo ships
  `tasks/drone_racer/` (env cfgs, MDP, track generator, agent cfgs) plus
  `model/{skrl,rsl_rl}/{train,play}.py`. Porting that stack would give a
  trainable racing benchmark in this repo. ~3,600 LoC + configs; deserves its
  own PR.
