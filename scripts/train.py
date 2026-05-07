import logging
import os
import random
import signal
import time
import yaml
import traceback

# Ensure the local omni_drones package (with student reward implementation)
# takes precedence over the system-installed one at /workspace/omni_drones/.
# This MUST happen before any `from omni_drones import ...` line, because
# Python caches the package path in sys.modules on first import.
import sys
import pathlib

FILE_PATH = os.path.dirname(__file__)
REPO_ROOT = pathlib.Path(FILE_PATH).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

# def set_global_reproducibility(seed: int, deterministic: bool = True):
#     """Seed common RNGs and opt into deterministic torch kernels."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # If the task yaml specifies a ppo_cfg field, load that file from cfg/algo/
    # and merge it over the default algo config, allowing per-task PPO settings.
    ppo_cfg_name = cfg.task.get("ppo_cfg", None)
    if ppo_cfg_name:
        import pathlib

        algo_dir = pathlib.Path(__file__).parent.parent / "cfg" / "algo"
        ppo_cfg_path = algo_dir / ppo_cfg_name
        if not ppo_cfg_path.suffix:
            ppo_cfg_path = ppo_cfg_path.with_suffix(".yaml")
        if ppo_cfg_path.exists():
            logging.info(f"Loading task-specific PPO config: {ppo_cfg_path}")
            task_ppo_cfg = OmegaConf.load(ppo_cfg_path)
            cfg.algo = OmegaConf.merge(cfg.algo, task_ppo_cfg)
        else:
            logging.warning(
                f"ppo_cfg '{ppo_cfg_name}' not found at {ppo_cfg_path}, using default."
            )

    # set_global_reproducibility(cfg.seed, deterministic=cfg.get("deterministic", True))

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    # Save config.yaml early to ensure it's available even if interrupted
    # Wandb saves config.yaml when finish() is called, but we want it saved immediately
    def save_config_early():
        try:
            config_path = os.path.join(run.dir, "files", "config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            # Save config in wandb's format: each key has a 'value' field
            config_dict = {}
            for key, value in run.config.items():
                config_dict[key] = {"value": value}
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Saved config.yaml to {config_path}")
        except Exception as e:
            logging.warning(f"Could not save config.yaml early: {e}")

    save_config_early()

    # Set up signal handler to ensure config is saved on Ctrl+C
    # The finally block will handle wandb.finish() and simulation_app.close()
    def signal_handler(sig, frame):
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = str(sig)
        logging.warning(
            f"Received signal {sig_name} ({sig}). Saving config and exiting..."
        )
        save_config_early()  # Save config immediately before cleanup
        # Let the exception propagate to trigger finally block
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation")
        )
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(
            base_env.observation_spec, ("agents", "observation_central")
        )
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "rate":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=rate requires cfg.task.drone_model.controller "
                    "to be set (e.g. RateController)."
                )
            transform = RateController(
                base_env.controller, action_key=("agents", "action")
            )
            transforms.append(transform)
        elif action_transform == "attitude":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=attitude requires cfg.task.drone_model.controller "
                    "to be set (e.g. AttitudeController)."
                )
            transform = AttitudeController(
                base_env.controller, action_key=("agents", "action")
            )
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int = 0, exploration_type: ExplorationType = ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v) for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        video_array = render_callback.get_video_array(axes="t c h w")
        if video_array is not None:
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), format="mp4"
            )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    loop_exception = None
    try:
        logging.info(
            f"Starting training loop: frames_per_batch={frames_per_batch}, "
            f"total_frames={total_frames}, num_envs={env.num_envs}"
        )
        pbar = tqdm(collector, total=total_frames // frames_per_batch)
        env.train()
        logging.info("Entering collector iteration.")
        for i, data in enumerate(pbar):
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            episode_stats.add(data.to_tensordict())

            if len(episode_stats) >= base_env.num_envs:
                stats = {
                    "train/"
                    + (".".join(k) if isinstance(k, tuple) else k): torch.mean(
                        v.float()
                    ).item()
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            info.update(policy.train_op(data.to_tensordict()))

            if eval_interval > 0 and i % eval_interval == 0:
                logging.info(f"Eval at {collector._frames} steps.")
                # info.update(evaluate(seed=cfg.seed))
                info.update(evaluate())
                env.train()
                base_env.train()

            if save_interval > 0 and i % save_interval == 0:
                try:
                    ckpt_path = os.path.join(
                        run.dir, f"checkpoint_{collector._frames}.pt"
                    )
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved checkpoint to {str(ckpt_path)}")
                except AttributeError:
                    logging.warning(
                        f"Policy {policy} does not implement `.state_dict()`"
                    )

            run.log(info)
            print(
                OmegaConf.to_yaml(
                    {k: v for k, v in info.items() if isinstance(v, float)}
                )
            )
            print(f"[train] epoch={i + 1} total_frames_processed={collector._frames}")

            pbar.set_postfix(
                {"rollout_fps": collector._fps, "frames": collector._frames}
            )

            if max_iters > 0 and i >= max_iters - 1:
                break

        logging.info("Collector iteration ended normally.")
        logging.info(f"Final Eval at {collector._frames} steps.")
        info = {"env_frames": collector._frames}
        # info.update(evaluate())
        # run.log(info)

        try:
            ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
            torch.save(policy.state_dict(), ckpt_path)

            model_artifact = wandb.Artifact(
                f"{cfg.task.name}-{cfg.algo.name.lower()}",
                type="model",
                description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
                metadata=dict(cfg),
            )

            model_artifact.add_file(ckpt_path)
            wandb.save(ckpt_path)
            run.log_artifact(model_artifact)

            logging.info(f"Saved checkpoint to {str(ckpt_path)}")
        except AttributeError:
            logging.warning(f"Policy {policy} does not implement `.state_dict()`")
    except BaseException as e:
        loop_exception = e
        logging.error(
            f"Training aborted with {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        raise
    finally:
        if loop_exception is None:
            logging.warning("Training loop reached cleanup without a caught exception.")
        else:
            logging.warning(
                f"Cleaning up after exception: {type(loop_exception).__name__}"
            )
        # Ensure config is saved and wandb is finished even on interruption
        save_config_early()
        wandb.finish()
        simulation_app.close()


if __name__ == "__main__":
    main()
