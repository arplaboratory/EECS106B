"""Smoke tests for the DreamerV3 / SkyDreamer integration.

These tests do NOT require IsaacSim — they only exercise the algo
registry, the env class import path, and (when JAX extras are installed)
the bridge's spec-derived attributes via a synthetic-spec construction.

Run from inside the distrobox / drone_venv:

    cd $EECS106B_DIR
    python3 -m pytest tests/test_dreamerv3_smoke.py -v

Or as a script:

    python3 tests/test_dreamerv3_smoke.py
"""
from __future__ import annotations

import importlib
import os
import sys

# Make sure the project root is on sys.path when run as a script.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def test_dreamerv3_in_algos_registry() -> None:
    """`dreamerv3` should appear in `omni_drones.learning.ALGOS` whenever the
    optional JAX backend imports cleanly. When it doesn't, the registry
    silently skips the entry — that path is also valid (PPO must keep working
    without JAX), so we accept either outcome but assert on the contract.
    """
    learning = importlib.import_module("omni_drones.learning")
    algos = learning.ALGOS
    if learning.DreamerV3Policy is None:
        # JAX extras absent — registry must still contain the standard algos.
        assert "ppo" in algos, "PPO must remain registered without JAX"
        assert "dreamerv3" not in algos
    else:
        assert "dreamerv3" in algos, "DreamerV3Policy imported but not in ALGOS"
        assert algos["dreamerv3"] is learning.DreamerV3Policy


def test_drone_race_dreamer_env_import() -> None:
    """The slim env subclass should import once IsaacSim is initialised.

    The parent ``DroneRaceEnv`` does ``import isaacsim.core.utils.prims`` at
    module-load time, which only resolves after ``init_simulation_app()`` has
    been called. When that hasn't happened (typical ``pytest`` run from a
    plain shell) we skip rather than fail — this isn't a regression in the
    dreamer code, it's the IsaacSim load-order requirement.
    """
    try:
        mod = importlib.import_module(
            "omni_drones.envs.drone_race.drone_race_dreamer"
        )
    except ModuleNotFoundError as e:
        if "isaacsim" in str(e) or "omni" in str(e):
            import pytest  # type: ignore[import-not-found]
            pytest.skip(f"IsaacSim not initialised in this process: {e}")
        raise
    assert hasattr(mod, "DroneRaceDreamerEnv")
    # Helper lifted in the same module.
    assert hasattr(mod, "make_dreamer_policy")


def test_dreamer_policy_constructs_against_synthetic_specs() -> None:
    """Build a `DreamerV3Policy` against synthetic torchrl specs and confirm
    spec-derived attributes are correct. Skipped when the JAX backend is
    absent (no point — `DreamerV3Policy` resolves to None in that case).
    """
    learning = importlib.import_module("omni_drones.learning")
    if learning.DreamerV3Policy is None:
        import pytest  # type: ignore[import-not-found]
        pytest.skip("DreamerV3Policy unavailable (install -e .[dreamer])")

    import torch
    from omegaconf import OmegaConf
    from torchrl.data import Composite, Unbounded

    num_envs = 4
    obs_dim, priv_dim, act_dim = 27, 15, 4
    device = torch.device("cpu")

    obs_spec = Composite({
        "agents": {
            "observation": Unbounded((1, obs_dim), device=device),
        },
        "info": {
            "drone_state": Unbounded((1, priv_dim), device=device),
        },
    }).expand(num_envs).to(device)
    act_spec = Composite({
        "agents": {"action": Unbounded((1, act_dim), device=device)},
    }).expand(num_envs).to(device)
    rew_spec = Composite({
        "agents": {"reward": Unbounded((1, 1), device=device)},
    }).expand(num_envs).to(device)

    cfg = OmegaConf.create({
        "name": "dreamerv3",
        "vision": False,
        "lambda_smooth": 0.0,
        "replay_context": 0,
        "disable_sharding": True,
        "batch_size": 4,
        "batch_length": 8,
        "imag_horizon": 4,
        "phase_schedule": {"batch_length_step": 0, "lr_step": 0},
    })

    policy = learning.DreamerV3Policy(
        cfg=cfg,
        observation_spec=obs_spec,
        action_spec=act_spec,
        reward_spec=rew_spec,
        device=device,
    )

    assert policy.obs_dim == obs_dim
    assert policy.priv_dim == priv_dim
    assert policy.action_dim == act_dim
    assert policy.image_shape is None  # vision=False
    assert policy.priv_key == ("info", "drone_state")
    assert policy.vision is False


if __name__ == "__main__":
    try:
        from _pytest.outcomes import Skipped  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        class Skipped(Exception):  # type: ignore[no-redef]
            pass

    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Skipped as exc:
                print(f"SKIP  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures.append((name, exc))
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    if failures:
        sys.exit(1)
    print("All smoke tests passed.")
