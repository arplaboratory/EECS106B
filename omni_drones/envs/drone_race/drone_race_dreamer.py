# MIT License
#
# Copyright (c) 2026 C. K. Wolfe
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


import logging

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from omni_drones.envs.drone_race.drone_race import DroneRaceEnv

# -----------------------------------------------------------------------------
# DreamerV3 / SkyDreamer policy import.
#
# The bridge in `omni_drones/learning/dreamerv3/policy.py` depends on the
# optional JAX backend (jax, optax, flax, elements, ninjax, portal) which
# only land via `pip install -e .[dreamer]`. We import it under a guard so
# this module still loads cleanly when those extras are absent — in that
# case `DreamerV3Policy` resolves to None and `make_dreamer_policy()` below
# raises a clean ImportError with the install hint, instead of crashing at
# module load and breaking unrelated PPO/SAC training.
# -----------------------------------------------------------------------------
try:
    from omni_drones.learning.dreamerv3 import DreamerV3Policy
except Exception as _dreamer_import_err:  # noqa: BLE001
    DreamerV3Policy = None
    logging.debug(
        "DreamerV3Policy unavailable in drone_race_dreamer: %s",
        _dreamer_import_err,
    )


class DroneRaceDreamerEnv(DroneRaceEnv):
    r"""
    Drone racing environment paired with the JAX-based DreamerV3 / SkyDreamer
    policy in ``omni_drones.learning.dreamerv3``. Subclasses
    :class:`DroneRaceEnv` so it inherits all gate placement, physics, reset
    and reward logic — its only role is to (a) document the obs / action
    contract that :class:`DreamerV3Policy` consumes and (b) expose the
    optional SkyDreamer hooks for the extended privileged channel and an
    image observation.

    ## Observation contract (consumed by DreamerV3Policy)

    | Key                          | Shape             | Required | Notes                                                                                                            |
    | ---------------------------- | ----------------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
    | ``("agents","observation")`` | ``(1, obs_dim)``  | yes      | Vector observation, inherited from ``DroneRaceEnv`` (15-dim drone state + 12-dim gate features).                 |
    | ``("agents","action")``      | ``(1, 4)``        | yes      | Body-rate command (thrust + 3 rates) — set by ``RateController``.                                                |
    | ``("info","drone_state")``   | ``(1, 15)``       | yes      | Privileged 15-dim drone state. Target of the SkyDreamer informed decoder. Provided by ``DroneRaceEnv``.          |
    | ``("info","priv")``          | ``(1, priv_dim)`` | no       | OPTIONAL extended privileged channel (paper :math:`o_t^+`). When present, the bridge prefers it over drone_state. |
    | ``("agents","image")``       | ``(1, 1, H, W)``  | no       | OPTIONAL image. Required only when both ``cfg.task.vision`` and ``cfg.algo.vision`` are ``true``.                |

    ## Reward  *(inherited from DroneRaceEnv)*

    The reward / termination logic in ``_compute_reward_and_done`` is
    inherited unchanged. Edit the parent class if you want different
    reward shaping for dreamer experiments.

    ## Launch

    Vector mode::

        python scripts/train.py task=DroneRaceDreamer algo=dreamerv3

    Vision mode (set the toggle on **both** task and algo)::

        python scripts/train.py task=DroneRaceDreamer task.vision=true \\
            algo=dreamerv3 algo.vision=true

    Install the optional JAX backend first::

        pip install -e .[dreamer]

    Without it the bridge runs in stub mode (zero actions, one-time warning).

    ## Programmatic usage (importing the dreamer policy)

    ::

        from hydra import compose, initialize
        from omni_drones.envs.drone_race import DroneRaceDreamerEnv
        from omni_drones.envs.drone_race.drone_race_dreamer import (
            make_dreamer_policy,
        )

        with initialize(config_path="../../../cfg", version_base=None):
            cfg = compose(config_name="train",
                          overrides=["task=DroneRaceDreamer",
                                     "algo=dreamerv3"])
        env = DroneRaceDreamerEnv(cfg, headless=True)
        policy = make_dreamer_policy(cfg, env)        # <- DreamerV3Policy

        td = env.reset()
        for _ in range(8):
            td = policy(td)                           # rollout-time inference
            td, _ = env.step(td)
        metrics = policy.train_op(td)                 # one world-model update

    ## Config

    | Parameter            | Type  | Default       | Description                                                            |
    | -------------------- | ----- | ------------- | ---------------------------------------------------------------------- |
    | ``vision``           | bool  | ``false``     | Adds the ``("agents","image")`` observation key.                       |
    | ``image_resolution`` | list  | ``[64, 64]``  | ``[H, W]`` for the image observation when ``vision`` is ``true``.      |

    All ``DroneRaceEnv`` config keys (``track_config``, ``gate_scale``,
    reward scales, etc.) apply unchanged.
    """

    # Reward / punishment scale keys we treat as "configured rewards"; if
    # every one of these is 0 (or absent) we assume the user is on the
    # default zeroed cfg and print the integration banner from __init__.
    _REWARD_KEYS_FOR_BANNER = (
        "reward_example",
        "reward_progress_scale",
        "reward_gate_passage",
        "reward_speed_scale",
        "reward_uprightness_scale",
        "rate_penalty_scale",
        "punishment_living",
        "punishment_truncated",
        "punishment_end_far",
        "punishment_ground_crash",
        "punishment_around_gate",
        "punishment_gate_crash",
        "punishment_backward",
        "corridor_velproj_scale",
    )

    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        if self._all_rewards_zero(cfg):
            self._print_integration_banner()

    @classmethod
    def _all_rewards_zero(cls, cfg) -> bool:
        for k in cls._REWARD_KEYS_FOR_BANNER:
            try:
                if float(cfg.task.get(k, 0.0)) != 0.0:
                    return False
            except Exception:  # noqa: BLE001
                continue
        return True

    @staticmethod
    def _print_integration_banner() -> None:
        banner = r"""
================================================================================
================================================================================
==                                                                            ==
==    ##  ##  ######  ##      ##       #####                                  ==
==    ##  ##  ##      ##      ##      ##   ##                                 ==
==    ######  ####    ##      ##      ##   ##                                 ==
==    ##  ##  ##      ##      ##      ##   ##                                 ==
==    ##  ##  ######  #####   #####    #####                                  ==
==                                                                            ==
================================================================================
==                                                                            ==
==  Everything integrates and runs. Good job launching the SkyDreamer         ==
==  pipeline -- IsaacSim, the JAX DreamerV3 bridge, and the env are all       ==
==  wired up correctly.                                                       ==
==                                                                            ==
==  Heads up: every reward / punishment scale is currently 0, so the          ==
==  policy has nothing to learn from. Set them up before a real run:          ==
==                                                                            ==
==    1. Reward scales:                                                       ==
==         cfg/task/DroneRaceDreamer.yaml                                     ==
==                                                                            ==
==    2. Reward computation (the actual reward formula):                      ==
==         omni_drones/envs/drone_race/drone_race.py                          ==
==         _compute_reward_and_done  (search 'STUDENT TODO 2/3')              ==
==                                                                            ==
==  This banner only fires while every scale is 0. As soon as you set         ==
==  any reward scale to a non-zero value it disappears.                       ==
==                                                                            ==
================================================================================
================================================================================
"""
        print(banner, flush=True)

    def _set_specs(self):
        # Inherit the base specs (drone_state, observation, action, reward,
        # stats are all set up by DroneRaceEnv._set_specs).
        super()._set_specs()

        # -----------------------------------------------------------------------
        # STUDENT TODO (1/2): Add `info.priv` — the SkyDreamer extended
        # privileged channel (paper o_t^+). Concatenate `drone_state` with
        # whatever extra task state you want the world model to reconstruct
        # (gate-relative poses, velocities, etc.). When present, the bridge
        # in `omni_drones/learning/dreamerv3/policy.py` auto-selects this
        # over `info.drone_state` as the informed-decoder target.
        #
        # Example:
        #   priv_dim = 15 + 6  # drone_state + e.g. next-gate rpos & yaw
        #   info_spec = Composite({
        #       "priv": Unbounded((1, priv_dim), device=self.device),
        #   }).expand(self.num_envs).to(self.device)
        #   self.observation_spec["info"]["priv"] = info_spec["priv"]
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # STUDENT TODO (2/2): Vision channel `agents.image`. Declares the
        # spec when `cfg.task.vision` is true; you still need to fill it in
        # `_compute_state_and_obs` below (currently emits zeros).
        # -----------------------------------------------------------------------
        if self.cfg.task.get("vision", False):
            h, w = self.cfg.task.get("image_resolution", [64, 64])
            image_spec = Composite({
                "image": Unbounded((1, 1, int(h), int(w)), device=self.device),
            }).expand(self.num_envs).to(self.device)
            self.observation_spec["agents"]["image"] = image_spec["image"]

    def _compute_state_and_obs(self) -> TensorDict:
        td = super()._compute_state_and_obs()

        # -----------------------------------------------------------------------
        # STUDENT TODO (1/2): Fill `info.priv` if you declared it above.
        #
        # priv = torch.cat([
        #     td[("info", "drone_state")],          # (N, 1, 15)
        #     # ... your extra privileged features here ...
        # ], dim=-1)
        # td.set(("info", "priv"), priv)
        # -----------------------------------------------------------------------

        # -----------------------------------------------------------------------
        # STUDENT TODO (2/2): Fill `agents.image`. The fallback below emits
        # zeros so the bridge has a tensor of the right shape; replace it
        # with one of:
        #   1. A geometric mask projected from gate corners (cheap, no
        #      Isaac sensor required).
        #   2. A real GateSegmask / Camera sensor: spawn it in
        #      `_design_scene` and read it via self._cam_sensor.get_images().
        # -----------------------------------------------------------------------
        if self.cfg.task.get("vision", False):
            h, w = self.cfg.task.get("image_resolution", [64, 64])
            zeros = torch.zeros(
                self.num_envs, 1, 1, int(h), int(w), device=self.device,
            )
            td.set(("agents", "image"), zeros)

        return td


# ---------------------------------------------------------------------------
# Dreamer policy wiring helper
# ---------------------------------------------------------------------------


def make_dreamer_policy(cfg, env):
    """Construct a :class:`DreamerV3Policy` from a :class:`DroneRaceDreamerEnv`.

    Mirrors the wiring used by ``scripts/train.py``: the env's specs and
    device determine the bridge's input / output shapes. ``cfg`` must be
    the full hydra cfg (so the policy can read ``cfg.algo`` and
    ``cfg.task``).

    Raises:
        ImportError: if the optional JAX backend is not installed.
            Run ``pip install -e .[dreamer]`` and retry.
    """
    if DreamerV3Policy is None:
        raise ImportError(
            "DreamerV3Policy could not be imported. Install the optional "
            "JAX backend with `pip install -e .[dreamer]`."
        )
    return DreamerV3Policy(
        cfg=cfg,
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        reward_spec=env.reward_spec,
        device=env.device,
    )
