# DreamerV3 / SkyDreamer integration

Companion notes for the JAX DreamerV3 backend in
`omni_drones/learning/dreamerv3/`. The README only documents the optional
`pip install -e .[dreamer]` step; everything else — launch commands, the
zero-reward integration banner, reward-design guidance, and references —
is collected here.

## Quick start

After `pip install -e .[dreamer]` (see README **Step 2b**) and inside the
usual distrobox + drone_venv shell:

```
python3 scripts/train.py task=DroneRaceDreamer algo=DroneRaceDreamer headless=true
```

This pairs `cfg/task/DroneRaceDreamer.yaml` (the env / track / reward
scales) with `cfg/algo/DroneRaceDreamer.yaml` (the full SkyDreamer
recipe — schedule, optimizer, loss scales, lambda_smooth, phase_schedule,
auto_stage). For a smaller / faster sanity-check cfg use
`algo=dreamerv3` instead.

The env class is `DroneRaceDreamerEnv` — a slim subclass of
`DroneRaceEnv` in `omni_drones/envs/drone_race/drone_race_dreamer.py`.
Extend it for the optional `info.priv` extended privileged channel or
`agents.image` vision input as documented in the file's docstring.

## Smoke verification

Before a long run, confirm the algo + env imports resolve cleanly:

```
python3 -c "from omni_drones.learning import ALGOS; print(sorted(ALGOS))"
# expected: [..., 'dreamerv3', 'happo', 'mappo', 'ppo', ...]

python3 -m pytest tests/test_dreamerv3_smoke.py -v
# expects 2 passed + 1 skipped (env-import test needs IsaacSim live).
```

If `dreamerv3` is missing from `ALGOS` the JAX import failed — re-check
the extras install and confirm `import jax` works in the same venv.

## Integration banner

When you launch with the default cfg, every reward / punishment scale in
`cfg/task/DroneRaceDreamer.yaml` is `0.0`, so the policy has no learning
signal. To make sure that's a deliberate choice rather than a missed
edit, `DroneRaceDreamerEnv.__init__` prints a large ASCII banner the
first time it is constructed under all-zero rewards:

```
================================================================================
==    ##  ##  ######  ##      ##       #####                                  ==
==    ######  ####    ##      ##      ##   ##                                 ==
==    ##  ##  ######  #####   #####    #####                                  ==
================================================================================
==  Everything integrates and runs. Good job launching the SkyDreamer ...
==  Heads up: every reward / punishment scale is currently 0 ...
================================================================================
```

The banner disappears as soon as **any** reward scale becomes non-zero;
no action item beyond `cfg/task/DroneRaceDreamer.yaml`.

## Reward design — *please prefer positive rewards*

> **Warning.** DreamerV3's value head uses a two-hot symlog
> distribution; in practice that means it can soak up large *positive*
> rewards but mishandles a regime that is dominated by large negative
> penalties. Empirically the actor collapses, imagination rollouts
> become unstable, and `dreamer/adv_std` plateaus near zero. Build your
> reward function out of additive **positive** shaping terms first
> (gate-progress, gate-passage, on-axis velocity), and only add penalty
> terms (rate / collision / out-of-bounds) once the positive terms
> dominate the typical episode return.

The SkyDreamer recipe in `cfg/algo/DroneRaceDreamer.yaml` follows this
pattern: `reward_progress_scale=5.0`, `reward_gate_passage=30.0`,
`rate_penalty_scale=1.0` (small relative to the gate reward), and the
crash punishments default to **0**. Mirror that ordering when filling
in your own scales in `cfg/task/DroneRaceDreamer.yaml`.

The actual reward formula lives in
`omni_drones/envs/drone_race/drone_race.py::_compute_reward_and_done`
under the `STUDENT TODO 2/3` block — that's where the per-step scalar
is assembled from the scales.

## Launch gotchas (non-interactive shells)

Inside an interactive `distrobox enter grasping` + `/opt/drone_venv`
session the launch command above just works. The two notes below only
matter from `docker exec` / non-login shells where Hydra's relative
config search path doesn't resolve.

1. **Pass `--config-dir` explicitly.** Hydra walks `file://../cfg`
   relative to `scripts/train.yaml`, which fails when the shell isn't
   the same one Hydra started in:

   ```
   python3 scripts/train.py \
     --config-dir $EECS106B_DIR/cfg --config-name=train \
     task=DroneRaceDreamer algo=DroneRaceDreamer headless=true
   ```

2. **Override under `task.env.*`, not `env.*`.** The top-level `env`
   is an interpolated alias of `task.env` and lives in a struct, so
   Hydra rejects new keys on it. Use `task.env.num_envs=4`:

   ```
   python3 scripts/train.py task=DroneRaceDreamer algo=DroneRaceDreamer \
     headless=true wandb.mode=disabled max_iters=2 task.env.num_envs=4
   ```

## References

- **DreamerV3** — Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse
  Domains through World Models.* arXiv:2301.04104 (2023).
  <https://arxiv.org/abs/2301.04104>
- **SkyDreamer** — drone-racing extension that this integration packages
  (informed decoder, smoothness regularizer, multi-phase schedule).
  *Add the SkyDreamer arxiv URL here once published.*
- Vendored upstream tree: `omni_drones/learning/dreamerv3/vendored/` —
  Hafner's reference `dreamerv3` + `embodied` packages, unmodified.
- Bridge implementation: `omni_drones/learning/dreamerv3/policy.py` —
  PyTorch / IsaacSim ↔ JAX adapter and SkyDreamer deltas.
