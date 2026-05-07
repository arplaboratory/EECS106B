# ----------------------------------------------------------------------------
# Copyright (c) 2026 C. K. Wolfe. All rights reserved.
#
# NOT FREE TO USE. The contents of this file (the PyTorch <-> JAX bridge that
# adapts the vendored dreamerv3 reference into the omni_drones training stack,
# along with the SkyDreamer-specific deltas implemented here) are the
# proprietary work of C. K. Wolfe. Redistribution, modification, or commercial
# use is not permitted without explicit written permission.
# ----------------------------------------------------------------------------

"""DreamerV3Policy

Bridge wrapper that lets the JAX-based dreamerv3 reference implementation
(vendored at omni_drones/learning/dreamerv3/vendored) be used as a drop-in
policy inside the existing PyTorch / torchrl / IsaacSim training loop in
scripts/train.py.

The contract this class fulfils — matching the same shape as PPOPolicy at
omni_drones/learning/ppo/ppo.py — is:

    DreamerV3Policy(cfg, observation_spec, action_spec, reward_spec, device)
        .__call__(td) -> td        # rollout-time inference
        .train_op(td) -> dict      # called once per collector batch
        .state_dict() / .load_state_dict()

Notes
-----
* The JAX backend is imported lazily on first use. If JAX (or any of the
  embodied / ninjax / optax dependencies) is missing, this class still
  imports cleanly but `__call__` / `train_op` will run in a degraded
  "stub" mode that emits zero actions and logs a one-time warning. Install
  the optional extras with `pip install -e .[dreamer]` to enable the real
  agent.
* SkyDreamer deltas implemented here:
    - Informed decoder: the decoder targets the privileged 15-dim drone state
      from `info.drone_state` rather than the raw observation.
    - Smoothness regularizer in imag_loss (patched in vendored agent.py) —
      coefficient is read from `cfg.algo.lambda_smooth`.
    - Multi-phase schedule: batch_length flips at `phase_schedule.batch_length_step`,
      learning rate / entropy at `phase_schedule.lr_step`.
* Vision toggle: when `cfg.algo.vision` is True the bridge expects an
  additional `("agents","image")` tensor in the obs TensorDict and exposes
  it to the agent's encoder under the key `image`.
"""

from __future__ import annotations

import io
import logging
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict


# Lazy backend import — keep the module importable without JAX.
_JAX_AVAILABLE: Optional[bool] = None
_jax = None
_jnp = None
_dreamerv3 = None
_embodied = None
_elements = None


def _try_import_backend():
    """Import jax/dreamerv3/embodied lazily; cache the result.

    The vendored ``embodied`` and ``dreamerv3`` packages use absolute imports
    (``import embodied``, ``import dreamerv3``) internally, so we prepend
    the vendored directory to ``sys.path`` before importing them.

    Sets XLA env vars BEFORE importing JAX so the JAX runtime doesn't
    pre-allocate the entire GPU memory pool — Isaac Sim co-resides on GPU 0
    and would otherwise be starved.
    """
    global _JAX_AVAILABLE, _jax, _jnp, _dreamerv3, _embodied, _elements
    if _JAX_AVAILABLE is not None:
        return _JAX_AVAILABLE
    try:
        import os
        # Cooperative GPU memory with Isaac Sim on GPU 0. Use the default BFC
        # allocator (much faster JIT compile than `platform`), but cap JAX's
        # share of each GPU so Isaac has headroom on GPU 0. PREALLOCATE=true
        # is the default and required for fast multi-GPU sharded compile.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
        import sys as _sys
        _vendored_dir = os.path.join(os.path.dirname(__file__), "vendored")
        if _vendored_dir not in _sys.path:
            _sys.path.insert(0, _vendored_dir)
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        import elements  # noqa: F401
        import dreamerv3 as _dv3  # noqa: F401  # from vendored/
        import embodied as _emb  # noqa: F401  # from vendored/

        _jax = jax
        _jnp = jnp
        _elements = elements
        _dreamerv3 = _dv3
        _embodied = _emb
        _JAX_AVAILABLE = True
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "[DreamerV3Policy] JAX backend unavailable (%s). Running in stub "
            "mode (zero actions). Install with `pip install -e .[dreamer]` "
            "to enable real DreamerV3 training.",
            e,
        )
        _JAX_AVAILABLE = False
    return _JAX_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t2np(x: torch.Tensor) -> np.ndarray:
    """torch -> numpy host bounce. dlpack zero-copy is left for a follow-up
    optimization; correctness first."""
    return x.detach().cpu().numpy()


def _np2t(x: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _torch_to_jax_dlpack(x: torch.Tensor):
    """torch.Tensor -> jax.Array via DLPack (zero-copy when both on CUDA).

    Falls back to host bounce if dlpack interop is unavailable.
    """
    try:
        import torch.utils.dlpack as _td
        cap = _td.to_dlpack(x.detach().contiguous())
        return _jax.dlpack.from_dlpack(cap)
    except Exception:
        return _jnp.asarray(x.detach().cpu().numpy())


def _jax_to_torch_dlpack(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """jax.Array -> torch.Tensor via DLPack (zero-copy when both on CUDA)."""
    try:
        import torch.utils.dlpack as _td
        cap = _jax.dlpack.to_dlpack(x)
        t = _td.from_dlpack(cap)
        if t.dtype != dtype:
            t = t.to(dtype)
        if t.device != device:
            t = t.to(device)
        return t
    except Exception:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _spec_shape(spec) -> tuple:
    """Best-effort extraction of a torchrl spec's per-env feature shape."""
    if hasattr(spec, "shape"):
        s = tuple(spec.shape)
        # Drop the leading batch dim (we're given the per-env spec usually).
        return s
    return ()


# ---------------------------------------------------------------------------
# Simple per-env sequence replay (used by the bridge instead of
# embodied.replay.Replay, which expects to be driven by embodied's full
# distributed loop).
# ---------------------------------------------------------------------------


class _SimpleSeqReplay:
    """Per-env circular buffer of step dicts; samples fixed-length sequences.

    Each step dict contains numpy arrays for every key in obs_keys, act_keys,
    and ext_keys. Capacity is per-env so the total buffer grows with num_envs.
    """

    def __init__(self, capacity_per_env, num_envs, length, obs_keys, act_keys, ext_keys, replay_context=0):
        self.length = int(length)
        self.num_envs = int(num_envs)
        self.capacity_per_env = int(capacity_per_env)
        self.obs_keys = list(obs_keys)
        self.act_keys = list(act_keys)
        self.ext_keys = list(ext_keys)
        # Tracked so sample() can do the right thing for is_first / consec
        # at the chunk boundary (the agent's _apply_replay_context branches
        # on `consec[:, 0] == 0` to choose between the warmed-up replay
        # carry and the previous batch's continuing carry).
        self.replay_context = int(replay_context)
        self.streams: list = [[] for _ in range(num_envs)]
        self._stepid_counter = 0

    def __len__(self):
        return sum(len(s) for s in self.streams)

    def add(self, env_idx: int, step: dict):
        s = self.streams[env_idx]
        s.append(step)
        if len(s) > self.capacity_per_env:
            del s[: len(s) - self.capacity_per_env]

    def can_sample(self) -> bool:
        return any(len(s) >= self.length for s in self.streams)

    def sample(self, batch_size: int):
        valid = [i for i, s in enumerate(self.streams) if len(s) >= self.length]
        if not valid:
            return None
        seqs = []
        for _ in range(batch_size):
            env_idx = int(np.random.choice(valid))
            stream = self.streams[env_idx]
            start = int(np.random.randint(0, len(stream) - self.length + 1))
            seqs.append(stream[start : start + self.length])

        # Stack into dict of (B, T, ...) numpy arrays.
        all_keys = list(self.obs_keys) + list(self.act_keys) + list(self.ext_keys)
        out: Dict[str, np.ndarray] = {}
        for k in all_keys:
            try:
                arr = np.stack(
                    [np.stack([step[k] for step in seq], axis=0) for seq in seqs],
                    axis=0,
                )
                out[k] = arr
            except KeyError:
                # Missing extra key: synthesize zeros of the right dtype.
                pass

        # Boundary handling differs by replay_context regime:
        # - replay_context == 0: no warmup carry available, so we set
        #   is_first[:, 0]=True to force the RSSM to reset cleanly at the
        #   chunk boundary (avoids leaking a stale train_carry from an
        #   unrelated sample into this sequence).
        # - replay_context > 0: the agent's _apply_replay_context branches
        #   on `consec[:, 0] == 0` to use the warmed-up replay carry built
        #   from the first K steps' dyn entries.  Each of our sampled
        #   chunks is independent, so we *always* want the replay carry —
        #   force consec[:, 0]=0.  We do NOT set is_first[:, 0]=True here:
        #   the warmup steps already supply correct dyn state, and an
        #   artificial reset would discard that warmup.
        if self.replay_context > 0:
            if "consec" in out:
                out["consec"][:, 0] = 0
        else:
            if "is_first" in out:
                out["is_first"][:, 0] = True
        return out


# ---------------------------------------------------------------------------
# Bridge policy
# ---------------------------------------------------------------------------


class DreamerV3Policy:
    """DreamerV3 policy adapter for the omni_drones training loop.

    See module docstring for full notes. The class deliberately tolerates a
    missing JAX backend by entering a stub mode (zero actions) so that the
    smoke test `python scripts/train.py task=DroneRaceDreamer algo=DroneRaceDreamer`
    can at least import this module and instantiate the policy.
    """

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device):
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.obs_key = ("agents", "observation")
        self.act_key = ("agents", "action")
        self.rew_key = ("next", "agents", "reward")
        # Prefer the SkyDreamer-style extended privileged channel `info.priv`
        # if the env exposes it (vision env does); fall back to the smaller
        # `info.drone_state` for the vector env.
        if ("info", "priv") in observation_spec.keys(True, True):
            self.priv_key = ("info", "priv")
        else:
            self.priv_key = ("info", "drone_state")
        self.image_key = ("agents", "image")

        # Hyperparameters
        self.train_every = int(cfg.get("train_every", 16))
        self.train_ratio = int(cfg.get("train_ratio", 128))
        self.batch_size = int(cfg.get("batch_size", 16))
        self.batch_length = int(cfg.get("batch_length", 64))
        self.replay_size = int(cfg.get("replay_size", 10_000_000))
        self.replay_context = int(cfg.get("replay_context", 16))
        self.imag_horizon = int(cfg.get("imag_horizon", 16))
        self.lambda_smooth = float(cfg.get("lambda_smooth", 0.0))
        self.informed_decode = bool(cfg.get("informed_decode", True))
        self.vision = bool(cfg.get("vision", False))
        self.model_size = str(cfg.get("model_size", "size12m"))

        ps = cfg.get("phase_schedule", {})
        self.phase_batch_length_step = int(ps.get("batch_length_step", 8_000_000))
        self.phase_batch_length_late = int(ps.get("batch_length_late", 256))
        self.phase_lr_step = int(ps.get("lr_step", 13_000_000))
        self.phase_lr_late = float(ps.get("lr_late", 2.0e-6))
        self.phase_entropy_late = float(ps.get("entropy_late", 1.0e-5))

        # Spec-derived shapes
        # observation_spec[("agents","observation")].shape is typically
        # (num_envs, 1, obs_dim); we want the per-step obs_dim.
        try:
            self.obs_dim = int(self.observation_spec[self.obs_key].shape[-1])
        except Exception:
            self.obs_dim = 0
        try:
            self.priv_dim = int(self.observation_spec[self.priv_key].shape[-1])
        except Exception:
            self.priv_dim = 0
        try:
            self.action_dim = int(self.action_spec[self.act_key].shape[-1])
        except Exception:
            self.action_dim = 0
        try:
            self.image_shape = tuple(self.observation_spec[self.image_key].shape[-3:])
        except Exception:
            self.image_shape = None  # vision off

        # Step bookkeeping (drives the multi-phase schedule).
        self.env_frames = 0
        self._last_action: Optional[torch.Tensor] = None
        self._is_first: Optional[torch.Tensor] = None  # set on first __call__

        # Auto-stage: paper's @13M lr/entropy drop, but triggered by
        # adv_std plateau detection rather than a fixed frame count.  The
        # symptom that motivates the transition (advantages collapse → entropy
        # term re-randomises the actor → gates_passed regresses) is what we
        # actually want to watch for, regardless of frame budget.
        from collections import deque as _deque
        self._auto_stage_enabled = bool(self.cfg.get("auto_stage", False))
        self._auto_stage_metric = str(self.cfg.get("auto_stage_metric", "adv_std"))
        self._auto_stage_threshold = float(self.cfg.get("auto_stage_threshold", 0.07))
        self._auto_stage_window = int(self.cfg.get("auto_stage_window", 5))
        self._auto_stage_min_frames = int(
            self.cfg.get("auto_stage_min_frames", 100_000)
        )
        self._auto_stage_history = _deque(maxlen=self._auto_stage_window)
        self._auto_stage_fired = False

        # JAX agent state (lazy).
        self._agent = None
        self._policy_carry = None
        self._train_carry = None
        self._replay = None
        self._stub_warned = False
        self._init_failed = False

        # Per-step queue of dyn/enc/dec entries returned by agent.policy()
        # at rollout time.  Populated in __call__ on every env step,
        # drained in _extend_replay so the entries land in the same chunk
        # slots as their (obs, act, reward).  Only used when
        # replay_context > 0 — otherwise the agent doesn't return entries.
        self._pending_entries: list = []

        if not _try_import_backend():
            logging.warning(
                "[DreamerV3Policy] Initialised in stub mode (no JAX). The "
                "policy will produce zero actions and `train_op` will be a no-op."
            )

    # ------------------------------------------------------------------
    # Lazy agent construction
    # ------------------------------------------------------------------

    def _build_agent_if_needed(self, num_envs: int):
        if self._agent is not None or self._init_failed:
            return
        if not _try_import_backend():
            return
        try:
            # Construct elements.Space-typed obs_space and act_space.
            Space = _elements.Space
            obs_space: Dict[str, Any] = {
                "obs": Space(np.float32, (self.obs_dim,)),
                "is_first": Space(bool, ()),
                "is_last": Space(bool, ()),
                "is_terminal": Space(bool, ()),
                "reward": Space(np.float32, ()),
            }
            # SkyDreamer informed-decode: include `priv` so the world-model
            # decoder learns to reconstruct privileged state alongside obs.
            if self.priv_dim > 0:
                obs_space["priv"] = Space(np.float32, (self.priv_dim,))
            if self.vision and self.image_shape is not None:
                # dreamerv3 Encoder expects images as uint8 HWC (it normalises
                # internally via /255). Our env emits CHW float in [0, 1]; the
                # bridge converts on each step. The spec advertised to the
                # agent uses HWC uint8.
                c, h, w = self.image_shape
                obs_space["image"] = Space(np.uint8, (h, w, c), 0, 256)

            act_space = {
                "action": Space(np.float32, (self.action_dim,), -1.0, 1.0),
            }
            self._obs_space_cache = obs_space
            self._act_space_cache = act_space

            # Build the flat dreamerv3 agent config (configs.yaml `defaults` +
            # `<model_size>` preset + our overrides + jax/logdir/seed/...).
            flat_config = self._build_dv3_config(num_envs=num_envs)

            from dreamerv3.agent import Agent as _DreamerAgent  # vendored
            self._agent = _DreamerAgent(obs_space, act_space, flat_config)

            # ext_space: extras that train() expects in `data` beyond obs/act
            # (consec, stepid, optionally enc/dyn/dec entries when
            # replay_context > 0). We force replay_context=0 in the bridge for
            # this first cut to avoid plumbing the encoder entries through.
            self._ext_space = dict(self._agent.model.ext_space)

            self._policy_carry = self._agent.init_policy(num_envs)
            self._train_carry = self._agent.init_train(self.batch_size)

            # Simple in-memory per-env sequence replay. The vendored
            # embodied.replay.Replay expects to be driven by the embodied
            # driver; using our own minimal buffer keeps timing under our
            # control inside the torchrl loop.
            # SDK convention (vendored/dreamerv3/main.py:183-187): the
            # buffer feeds chunks of `batch_length + replay_context` records.
            # The agent's _apply_replay_context slices the first
            # `replay_context` steps off as RSSM warmup, then runs the loss
            # on the remaining `batch_length` positions.  agent_cfg's
            # batch_length is the *loss segment length*, NOT the chunk
            # length — so JIT compiles for (B, batch_length, ...) input
            # but expects to receive (B, batch_length + replay_context, ...)
            # before the slice.
            chunk_length = self.batch_length + self.replay_context
            self._replay = _SimpleSeqReplay(
                capacity_per_env=max(
                    self.replay_size // max(num_envs, 1),
                    8 * chunk_length,
                ),
                num_envs=num_envs,
                length=chunk_length,
                obs_keys=list(obs_space.keys()),
                act_keys=list(act_space.keys()),
                ext_keys=list(self._ext_space.keys()),
                replay_context=self.replay_context,
            )
            self._num_envs = num_envs
            logging.info(
                "[DreamerV3Policy] dreamerv3 agent built "
                "(num_envs=%d, obs_dim=%d, priv_dim=%d, act_dim=%d, vision=%s, "
                "batch=%dx%d).",
                num_envs, self.obs_dim, self.priv_dim, self.action_dim,
                self.vision, self.batch_size, self.batch_length,
            )
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            logging.warning(
                "[DreamerV3Policy] Failed to build dreamerv3 agent (%s). "
                "Falling back to stub mode.\n%s", e, _tb.format_exc(),
            )
            self._init_failed = True
            self._agent = None

    def _build_dv3_config(self, num_envs: int = 1):
        """Build the flat elements.Config object the dreamerv3 Agent expects.

        Mirrors the canonical loading pattern in vendored/dreamerv3/main.py:
            configs = yaml.load(configs.yaml)
            config = elements.Config(configs['defaults'])
            config = config.update(configs[<size_preset>])
            <apply our overrides>
            agent_cfg = elements.Config(
                **config.agent, jax=config.jax, logdir=..., seed=...,
                batch_size=..., batch_length=..., replay_context=...,
                report_length=..., replica=0, replicas=1)
        """
        import os
        import pathlib
        import tempfile
        import ruamel.yaml as _yaml
        cfg_path = (
            pathlib.Path(__file__).parent / "vendored" / "dreamerv3" / "configs.yaml"
        )
        with open(cfg_path, "r") as f:
            configs = _yaml.YAML(typ="safe").load(f)

        config = _elements.Config(configs["defaults"])
        if self.model_size in configs:
            config = config.update(configs[self.model_size])

        # Targeted overrides into the `agent.*` and `jax.*` subtrees.
        overrides = {
            "agent.imag_loss.lam": float(self.cfg.get("return_lambda", 0.95)),
            "agent.imag_loss.actent": float(self.cfg.get("entropy_coef", 3e-4)),
            "agent.imag_loss.smooth": float(self.lambda_smooth),
            "agent.imag_loss.slowtar": bool(self.cfg.get("slowtar", True)),
            "agent.imag_loss.slowreg": float(self.cfg.get("slowreg", 1.0)),
            "agent.horizon": int(round(
                1.0 / max(1e-6, 1.0 - float(self.cfg.get("gamma", 0.997)))
            )),
            "agent.opt.lr": float(self.cfg.get("lr", 4e-5)),
            "agent.opt.agc": float(self.cfg.get("agc", 0.3)),
            "agent.opt.eps": float(self.cfg.get("eps", 1e-20)),
            "agent.opt.beta1": float(self.cfg.get("beta1", 0.9)),
            "agent.opt.beta2": float(self.cfg.get("beta2", 0.999)),
            # Disable the report path — we don't drive embodied's report loop.
            "agent.report": False,
            # Don't precompile (faster startup).
            "jax.expect_devices": 0,
        }
        ls = self.cfg.get("loss_scales", None)
        if ls is not None:
            for k, v in dict(ls).items():
                overrides[f"agent.loss_scales.{k}"] = float(v)

        # Multi-GPU sharding: use every JAX-visible device for both the policy
        # and train passes when both batch_size and num_envs divide evenly.
        # dreamerv3 uses PartitionSpec(('d','f')) on the leading axis, so a
        # (n_devs,1,1) mesh requires:
        #   batch_size % n_devs == 0   (train pass)
        #   num_envs   % n_devs == 0   (init_policy / policy pass)
        # If either constraint fails we keep the agent on a single device
        # (GPU 0). The trainer is rarely the actual bottleneck (Isaac Sim env
        # step usually is), so lost trainer parallelism is small.
        try:
            n_devs = int(_jax.device_count())
        except Exception:
            n_devs = 1
        # Opt-out: cfg.algo.disable_sharding=true forces single-device JAX even
        # when multiple GPUs are visible (useful when the multi-device JIT
        # compile is too slow for the run budget).
        if bool(self.cfg.get("disable_sharding", False)):
            logging.info("[DreamerV3Policy] disable_sharding=true -> "
                         "forcing single-device JAX (device count was %d).", n_devs)
            n_devs = 1
        bs_ok = int(self.batch_size) % n_devs == 0
        ne_ok = int(num_envs) % n_devs == 0
        if n_devs > 1 and bs_ok and ne_ok:
            dev_list = list(range(n_devs))
            overrides["jax.policy_devices"] = dev_list
            overrides["jax.train_devices"] = dev_list
            overrides["jax.policy_mesh"] = [n_devs, 1, 1]
            overrides["jax.train_mesh"] = [n_devs, 1, 1]
            logging.info(
                "[DreamerV3Policy] sharding across %d devices "
                "(batch_size=%d, num_envs=%d).",
                n_devs, self.batch_size, num_envs,
            )
        elif n_devs > 1:
            logging.info(
                "[DreamerV3Policy] %d JAX devices visible but batch_size=%d "
                "(div=%s) and num_envs=%d (div=%s); keeping single-device mesh.",
                n_devs, self.batch_size, bs_ok, num_envs, ne_ok,
            )

        for k, v in overrides.items():
            try:
                config = config.update({k: v})
            except Exception:
                pass

        # Logdir must exist (the embodied.jax.Agent writes there).
        logdir = os.path.join(tempfile.gettempdir(), "dreamerv3_omni_logs")
        os.makedirs(logdir, exist_ok=True)

        # SkyDreamer §III-A explicitly tunes replay_context=16 (DV3 default
        # is 1).  With it set, the agent's `_apply_replay_context` uses the
        # first K=replay_context steps of every sampled chunk to warm-start
        # the RSSM dyn carry from the per-step `dyn/deter`+`dyn/stoch`
        # entries that were written to the buffer at rollout time.  Loss is
        # then computed only on positions [K:T], so each chunk has K steps
        # of warmup + (batch_length-K) steps that contribute to gradients.
        # Encoder + Decoder have empty entry_space (rssm.py:200-208 and
        # :278-286) so the only carry plumbed through the buffer is
        # `dyn/deter` and `dyn/stoch`.
        agent_cfg = _elements.Config(
            **config.agent,
            logdir=logdir,
            seed=int(self.cfg.get("seed", 0)),
            jax=config.jax,
            batch_size=int(self.batch_size),
            batch_length=int(self.batch_length),
            replay_context=int(self.replay_context),
            report_length=int(config.report_length),
            replica=0,
            replicas=1,
        )
        return agent_cfg

    # ------------------------------------------------------------------
    # Rollout-time inference
    # ------------------------------------------------------------------

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # The collector calls this at every env step. Output shape for the
        # action key matches the action_spec shape: (num_envs, 1, action_dim).
        obs = tensordict.get(self.obs_key)
        num_envs = obs.shape[0]

        # First-time init.
        self._build_agent_if_needed(num_envs)

        # Track is_first using terminated/truncated from the previous step's
        # `next` block when the collector populates it; on the very first call
        # set everything to True.
        if self._is_first is None:
            is_first = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        else:
            is_first = self._is_first
        self._is_first = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        if self._agent is None:
            # Stub: zero action, correct shape.
            zeros = torch.zeros(*self.action_spec[self.act_key].shape,
                                device=self.device)
            tensordict.set(self.act_key, zeros)
            if not self._stub_warned:
                logging.warning("[DreamerV3Policy] Producing zero actions (stub mode).")
                self._stub_warned = True
            return tensordict

        try:
            obs_dict = self._pack_obs(tensordict, is_first)
            # SkyDreamer paper §II-G: at inference / eval the policy is
            # deterministic (σ=0). We map our env's evaluation phase to
            # `agent.policy(mode='eval')` which returns the distribution mode
            # (μ for a Gaussian) instead of a sample. Detection: torchrl's
            # exploration mode -- when the env is in eval / no-exploration
            # mode the collector sets exploration to MODE/MEAN. We can
            # also force this via the `eval_deterministic` algo flag.
            policy_mode = "train"
            if bool(self.cfg.get("eval_deterministic", True)):
                try:
                    from torchrl.envs.utils import exploration_type, ExplorationType
                    et = exploration_type()
                    if et == ExplorationType.MODE or et == ExplorationType.MEAN:
                        policy_mode = "eval"
                except Exception:
                    pass
            self._policy_carry, act_dict, policy_outs = self._agent.policy(
                self._policy_carry, obs_dict, mode=policy_mode
            )
            # When replay_context > 0, agent.policy() returns dyn/enc/dec
            # entries in `policy_outs` (agent.py:127-134).  Capture them per
            # step so _extend_replay can pack them into the buffer alongside
            # (obs, act, reward).  Encoder + Decoder have empty entry_space
            # so in practice only `dyn/deter` and `dyn/stoch` show up here.
            if self.replay_context > 0 and policy_outs is not None:
                step_entries = {}
                for k, v in policy_outs.items():
                    if k.startswith(("enc/", "dyn/", "dec/")):
                        step_entries[k] = np.asarray(v)
                self._pending_entries.append(step_entries)
            # Stash the previous-step act for replay packing; act_dict values
            # are numpy arrays of shape (num_envs, action_dim).
            # Bridge fix #4: dreamerv3's `bounded_normal` policy returns
            # `Normal(tanh(mean), stddev)` and does NOT clip samples; rare
            # samples can fall outside [-1, 1]. Hard-clip here so the env's
            # RateController never sees out-of-range commands.
            act_np = np.clip(np.asarray(act_dict["action"]), -1.0, 1.0)
            self._last_action_np = act_np
            self._last_obs_dict = obs_dict
            act = _np2t(act_np, self.device, dtype=torch.float32)
            # Match action spec shape: (num_envs, 1, action_dim)
            target_shape = self.action_spec[self.act_key].shape
            if act.shape != tuple(target_shape):
                act = act.reshape(target_shape)
            tensordict.set(self.act_key, act)
        except Exception as e:  # noqa: BLE001
            import traceback as _tb
            logging.warning(
                "[DreamerV3Policy] policy() failed (%s); zero action.\n%s",
                e, _tb.format_exc(),
            )
            zeros = torch.zeros(*self.action_spec[self.act_key].shape,
                                device=self.device)
            tensordict.set(self.act_key, zeros)
        return tensordict

    def _pack_obs(self, tensordict: TensorDict, is_first: torch.Tensor) -> Dict[str, Any]:
        obs = tensordict.get(self.obs_key)  # (N, 1, D)
        # Squeeze agent dim.
        if obs.dim() >= 3 and obs.shape[1] == 1:
            obs = obs.squeeze(1)
        priv_t = tensordict.get(self.priv_key, default=None)
        if priv_t is None:
            priv_t = torch.zeros(obs.shape[0], self.priv_dim, device=self.device)
        elif priv_t.dim() >= 3 and priv_t.shape[1] == 1:
            priv_t = priv_t.squeeze(1)

        obs_dict: Dict[str, Any] = {
            "obs": _t2np(obs.float()),
            "priv": _t2np(priv_t.float()),
            "is_first": _t2np(is_first.bool()),
            "is_last": np.zeros((obs.shape[0],), dtype=bool),
            "is_terminal": np.zeros((obs.shape[0],), dtype=bool),
            "reward": np.zeros((obs.shape[0],), dtype=np.float32),
        }
        if self.vision:
            img = tensordict.get(self.image_key, default=None)
            if img is not None:
                # Env emits CHW float32 in [0, 1] with shape (N, 1, C, H, W).
                # Convert to (N, H, W, C) uint8 for the dreamerv3 encoder.
                if img.dim() == 5 and img.shape[1] == 1:
                    img = img.squeeze(1)
                # CHW -> HWC, scale to [0, 255], cast to uint8
                img = img.permute(0, 2, 3, 1).clamp(0.0, 1.0).mul_(255.0)
                obs_dict["image"] = img.to(torch.uint8).cpu().numpy()
        return obs_dict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_op(self, data: TensorDict) -> Dict[str, float]:
        # Update the env_frames counter (drives the phase schedule).
        if data.batch_size:
            self.env_frames += int(np.prod(tuple(data.batch_size)))

        # Apply phase schedule transitions.
        self._maybe_apply_phase_schedule()

        # Update is_first for next rollout step from this batch's terminations.
        try:
            term = data.get(("next", "terminated"), default=None)
            trunc = data.get(("next", "truncated"), default=None)
            if term is not None or trunc is not None:
                last = data.get(("next", "done"), default=None)
                if last is None and (term is not None or trunc is not None):
                    a = term if term is not None else trunc
                    b = trunc if trunc is not None else term
                    last = (a.bool() | b.bool())
                if last is not None:
                    # Squeeze trailing dims to (num_envs,).
                    last = last.reshape(last.shape[0], -1).any(dim=-1)
                    self._is_first = last.to(self.device)
        except Exception:
            pass

        if self._agent is None:
            return {"dreamer/stub": 1.0}

        # Append rollout to replay (per-env streams).
        try:
            self._extend_replay(data)
        except Exception as e:  # noqa: BLE001
            logging.debug("[DreamerV3Policy] replay extend failed: %s", e)

        if self._replay is None or not self._replay.can_sample():
            return {"dreamer/replay_size": float(0 if self._replay is None
                                                 else len(self._replay))}

        # Number of gradient steps for this batch.
        steps = int(self.train_ratio * self.train_every)
        # During smoke testing we cap this so training isn't pathological at
        # tiny num_envs. Real training should let it run.
        smoke_cap = int(self.cfg.get("max_grad_steps_per_batch", 0))
        if smoke_cap > 0:
            steps = min(steps, smoke_cap)

        metrics_accum: Dict[str, float] = {}
        # Place batch entries on device with the right sharding so the agent's
        # JIT-compiled train() doesn't trigger "Disallowed host-to-device
        # transfer" errors (XLA flag forbids implicit transfers).
        train_sharded = self._agent.train_sharded
        train_mirrored = self._agent.train_mirrored
        for step_i in range(steps):
            try:
                batch_np = self._replay.sample(self.batch_size)
                if batch_np is None:
                    break
                # Move each leaf to device with the appropriate sharding.
                batch_dev = {
                    k: _jax.device_put(v, train_sharded)
                    for k, v in batch_np.items()
                }
                # Seed: use the agent's helper so it gets train_mirrored.
                batch_dev["seed"] = self._agent._seeds(
                    self._agent.n_updates.value + step_i,
                    train_mirrored,
                )
                # Bridge audit fix #6: zero the prevact slot of `train_carry`
                # at the boundary of each batch.  When `replay_context=0`,
                # `_apply_replay_context` prepends `carry[3]` (the last action
                # stored in the carry tuple) onto `data[act][:, :-1]`.  Since
                # we share train_carry across unrelated sampled sequences,
                # this would otherwise leak the previous batch's last action
                # into the current sequence.  Replay sample also forces
                # is_first[:, 0]=True so the RSSM resets cleanly.
                try:
                    self._train_carry = self._reset_prevact(self._train_carry)
                except Exception:
                    pass
                self._train_carry, _outs, mets = self._agent.train(
                    self._train_carry, batch_dev
                )
                for k, v in mets.items():
                    try:
                        metrics_accum[f"dreamer/{k}"] = float(np.asarray(v).mean())
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                logging.warning(
                    "[DreamerV3Policy] train step failed (%s):\n%s",
                    e, _tb.format_exc(),
                )
                break

        metrics_accum["dreamer/env_frames"] = float(self.env_frames)
        metrics_accum["dreamer/replay_size"] = float(len(self._replay))

        # Auto-stage: track the chosen metric, fire phase transition on plateau.
        if self._auto_stage_enabled and not self._auto_stage_fired:
            v = metrics_accum.get(f"dreamer/{self._auto_stage_metric}", None)
            if v is not None:
                self._auto_stage_history.append(float(v))
            self._maybe_apply_auto_stage(metrics_accum)
        return metrics_accum

    def _maybe_apply_auto_stage(self, metrics: Dict[str, float]):
        """Fire the paper's lr/entropy drop when the chosen metric plateaus.

        Default trigger: ``adv_std < threshold`` for the entire rolling window
        AND ``env_frames >= min_frames``. Once fired, the optimizer is rebuilt
        with ``phase_lr_late`` and ``phase_entropy_late`` (paper §III-A).
        """
        if self._auto_stage_fired or self._agent is None:
            return
        if self.env_frames < self._auto_stage_min_frames:
            return
        if len(self._auto_stage_history) < self._auto_stage_window:
            return
        hist = list(self._auto_stage_history)
        if max(hist) >= self._auto_stage_threshold:
            return  # not plateaued yet
        try:
            current_lr = float(self.cfg.get("lr", 4e-5))
            current_ent = float(self.cfg.get("entropy_coef", 3e-4))
            logging.info(
                "[DreamerV3Policy] AUTO-STAGE FIRED: %s plateaued <%.3f for %d "
                "iters at frame=%d. lr %g -> %g, entropy %g -> %g.",
                self._auto_stage_metric, self._auto_stage_threshold,
                self._auto_stage_window, self.env_frames,
                current_lr, self.phase_lr_late,
                current_ent, self.phase_entropy_late,
            )
            self.cfg.lr = self.phase_lr_late
            self.cfg.entropy_coef = self.phase_entropy_late
            self._rebuild_optimizer(
                lr=self.phase_lr_late,
                entropy_coef=self.phase_entropy_late,
            )
            self._auto_stage_fired = True
            metrics["dreamer/auto_stage_fired"] = 1.0
        except Exception as e:  # noqa: BLE001
            logging.warning("[DreamerV3Policy] auto-stage rebuild failed: %s", e)

    def _extend_replay(self, data: TensorDict):
        """Push the current rollout into the per-env replay streams.

        data is a TensorDict shaped (num_envs, T, ...). We split along the env
        axis and feed sequential transitions, building per-step dicts whose
        keys match obs_space + act_space + ext_space.
        """
        if self._replay is None or self._agent is None:
            return
        obs = data.get(self.obs_key)
        priv = data.get(self.priv_key, default=None)
        act = data.get(self.act_key)
        rew = data.get(self.rew_key, default=None)
        term = data.get(("next", "terminated"), default=None)
        trunc = data.get(("next", "truncated"), default=None)
        img = data.get(self.image_key, default=None)
        N = obs.shape[0]
        T = obs.shape[1]
        # ext_space: consec (int32 scalar), stepid (uint8 of size 20)
        # Bridge fix #1, #2, #5: episode-aware is_first / is_last / is_terminal.
        # `is_first` should be True for the very first step in the chunk AND
        # for any step that immediately follows a terminate-or-truncate. The
        # collector hands us `("next","terminated"|"truncated")` per step, so
        # `is_first[t] = (t==0) or term_or_trunc[t-1]`.  `is_last` should mark
        # the chunk end AND any step where this slot terminates / truncates,
        # so the world model breaks the bootstrap there. `is_terminal` stays
        # term-only (truncation should NOT zero the bootstrap return).
        if term is not None:
            term_np = term.bool().reshape(N, T, -1).any(dim=-1).cpu().numpy()
        else:
            term_np = np.zeros((N, T), dtype=bool)
        if trunc is not None:
            trunc_np = trunc.bool().reshape(N, T, -1).any(dim=-1).cpu().numpy()
        else:
            trunc_np = np.zeros((N, T), dtype=bool)
        last_or_term = term_np | trunc_np  # (N, T)

        # Drain T per-step rollout entries from the queue.  Each entry is a
        # dict {key: ndarray (num_envs, ...)}.  In normal operation the queue
        # has exactly T items (one per __call__ in this rollout cycle); on
        # mismatch (e.g. recovery from a failed step) we use min(T, available)
        # and skip entry plumbing for steps without a matching policy output.
        if self.replay_context > 0 and self._pending_entries:
            entries_chunk = self._pending_entries[:T]
            self._pending_entries = self._pending_entries[T:]
        else:
            entries_chunk = []

        for env_idx in range(N):
            for t in range(T):
                if t == 0:
                    is_first_val = True
                else:
                    is_first_val = bool(last_or_term[env_idx, t - 1])
                is_last_val = bool(last_or_term[env_idx, t]) or (t == T - 1)
                is_terminal_val = bool(term_np[env_idx, t])

                step = {
                    "obs": _t2np(obs[env_idx, t].reshape(-1).float()),
                    "action": _t2np(act[env_idx, t].reshape(-1).float()),
                    "reward": np.float32(
                        rew[env_idx, t].sum().item() if rew is not None else 0.0
                    ),
                    "is_first": np.bool_(is_first_val),
                    "is_last": np.bool_(is_last_val),
                    "is_terminal": np.bool_(is_terminal_val),
                }
                if priv is not None and self.priv_dim > 0:
                    step["priv"] = _t2np(priv[env_idx, t].reshape(-1).float())
                if self.vision and img is not None:
                    # img[env_idx, t] is shape (1, C, H, W) (agent dim leading).
                    img_t = img[env_idx, t]
                    if img_t.dim() == 4 and img_t.shape[0] == 1:
                        img_t = img_t.squeeze(0)
                    # CHW float [0, 1] -> HWC uint8 to match obs_space["image"].
                    img_hwc = img_t.permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0)
                    step["image"] = img_hwc.to(torch.uint8).cpu().numpy()
                # ext keys
                if "consec" in self._ext_space:
                    step["consec"] = np.int32(t)
                if "stepid" in self._ext_space:
                    sid = np.zeros(20, dtype=np.uint8)
                    sid[: 4] = np.frombuffer(
                        np.uint32(self._replay._stepid_counter).tobytes(),
                        dtype=np.uint8,
                    )
                    step["stepid"] = sid
                    self._replay._stepid_counter += 1
                # Plumbed-through dyn entries (shape (num_envs, ...) per step
                # → per-env (...) here).  Encoder/decoder entries fall here too
                # but with empty value dicts they're skipped.
                if t < len(entries_chunk):
                    for k, v in entries_chunk[t].items():
                        step[k] = v[env_idx]
                try:
                    self._replay.add(env_idx, step)
                except Exception as e:  # noqa: BLE001
                    logging.debug("[DreamerV3Policy] replay add failed: %s", e)
                    return

    def _reset_prevact(self, carry):
        """Zero the prevact slot of an init_train/init_policy carry tuple."""
        if not isinstance(carry, tuple) or len(carry) < 4:
            return carry
        prevact = carry[-1]
        if isinstance(prevact, dict):
            zero_prevact = {k: _jnp.zeros_like(v) for k, v in prevact.items()}
        else:
            zero_prevact = _jnp.zeros_like(prevact)
        return (*carry[:-1], zero_prevact)

    def _maybe_apply_phase_schedule(self):
        """Bake the SkyDreamer multi-phase schedule into the running run.

        At ``phase_batch_length_step`` we flip ``batch_length`` 64 -> 256.
        At ``phase_lr_step`` we drop ``lr`` 4e-5 -> 2e-6 and ``entropy_coef``
        3e-4 -> 1e-5. Both phases ALSO rebuild the agent's optimizer so the
        new lr / entropy actually take effect (logging alone was a TODO).
        """
        if self._agent is None:
            return
        if (
            self.env_frames >= self.phase_batch_length_step
            and self.batch_length != self.phase_batch_length_late
        ):
            new_bl = self.phase_batch_length_late
            logging.info("[DreamerV3Policy] Phase: batch_length %d -> %d at %d frames",
                         self.batch_length, new_bl, self.env_frames)
            self.batch_length = new_bl
            try:
                if self._replay is not None:
                    # Buffer feeds chunks of (batch_length + replay_context)
                    # records — agent slices off `replay_context` for warmup,
                    # leaves `batch_length` for the loss segment.
                    self._replay.length = new_bl + self.replay_context  # type: ignore[attr-defined]
            except Exception:
                pass
        if (
            self.env_frames >= self.phase_lr_step
            and float(self.cfg.get("lr", 4e-5)) != self.phase_lr_late
        ):
            logging.info("[DreamerV3Policy] Phase: lr/entropy late switch "
                         "(lr=%g entropy=%g) at %d frames",
                         self.phase_lr_late, self.phase_entropy_late,
                         self.env_frames)
            try:
                self.cfg.lr = self.phase_lr_late
                self.cfg.entropy_coef = self.phase_entropy_late
                self._rebuild_optimizer(
                    lr=self.phase_lr_late,
                    entropy_coef=self.phase_entropy_late,
                )
            except Exception as e:  # noqa: BLE001
                logging.warning("[DreamerV3Policy] phase optimizer rebuild "
                                "failed: %s", e)

    def _rebuild_optimizer(self, lr: float, entropy_coef: float):
        """Rebuild the agent's optax optimizer with the new lr / entropy.

        dreamerv3.agent.Agent constructs ``self.opt = embodied.jax.Optimizer(
            self.modules, self._make_opt(**config.opt), summary_depth=1,
            name='opt')`` once at __init__; the only mutable state is the
        running optax momentum which is fine to reset at a phase boundary
        since the policy is changing anyway.  Updating ``actent`` simply
        means changing the value the imag_loss reads.
        """
        from importlib import import_module as _im
        embodied_jax = _im("embodied.jax")
        model = self._agent.model
        # Update the in-memory config the agent reads each step.
        try:
            model.config = model.config.update({
                "opt.lr": float(lr),
                "imag_loss.actent": float(entropy_coef),
            })
        except Exception:
            pass
        # Replace the optimizer module.  embodied.jax.Optimizer caches its
        # optax state, so this resets momentum etc. — acceptable mid-training.
        new_opt = embodied_jax.Optimizer(
            model.modules, model._make_opt(**model.config.opt),
            summary_depth=1, name="opt",
        )
        model.opt = new_opt

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self._agent is None:
            return {"dreamer_stub": True, "env_frames": self.env_frames}
        try:
            buf = io.BytesIO()
            pickle.dump(self._agent.save(), buf)
            agent_bytes = torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8))
            return {
                "agent_bytes": agent_bytes,
                "env_frames": self.env_frames,
                "batch_length": self.batch_length,
            }
        except Exception as e:  # noqa: BLE001
            logging.warning("[DreamerV3Policy] state_dict() failed: %s", e)
            return {"env_frames": self.env_frames}

    def load_state_dict(self, sd: Dict[str, Any]):
        self.env_frames = int(sd.get("env_frames", 0))
        self.batch_length = int(sd.get("batch_length", self.batch_length))
        if self._agent is not None and "agent_bytes" in sd:
            try:
                buf = io.BytesIO(sd["agent_bytes"].numpy().tobytes())
                state = pickle.load(buf)
                self._agent.load(state)
            except Exception as e:  # noqa: BLE001
                logging.warning("[DreamerV3Policy] load_state_dict failed: %s", e)
