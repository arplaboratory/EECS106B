import os
import time

import hydra
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_drones import init_simulation_app
import omni_drones.envs

FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="mpc", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    import omni_drones.envs.drone_race

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)

    render_interval_sec = cfg.get("render_interval", 0.05)
    policy_dt = env.cfg.sim.dt * env.substeps
    render_every_n_steps = max(1, round(render_interval_sec / policy_dt))
    print(f"render_every_n_steps: {render_every_n_steps}")
    print(f"render_interval_sec: {render_interval_sec}")
    print(f"policy_dt: {policy_dt}")

    step_counter = [0]
    substeps = env.substeps

    def _render_fn(substep: int) -> bool:
        if substep == substeps - 1:
            step_counter[0] += 1
        return substep == substeps - 1 and (step_counter[0] % render_every_n_steps == 0)

    env.enable_render(_render_fn)
    env.set_seed(cfg.seed)

    if not hasattr(env, "compute_mpc_action"):
        raise AttributeError(
            f"{type(env).__name__} does not implement compute_mpc_action()."
        )

    td = env.reset()
    step = 0
    ep_step = 0
    stats_every = int(cfg.get("stats_every", 50))
    wall_run0 = time.perf_counter()
    wall_ep0 = wall_run0
    prev_track_done = False

    while simulation_app.is_running():
        actions = env.compute_mpc_action()
        action_td = TensorDict(
            {"agents": {"action": actions}},
            batch_size=env.batch_size,
            device=env.device,
        )
        td = env.step(action_td)
        next_td = td["next"]
        step += 1
        ep_step += 1
        sim_ep_t = ep_step * policy_dt
        wall_run_t = time.perf_counter() - wall_run0

        if step % max(1, stats_every) == 0:
            reward = next_td["agents", "reward"].mean().item()
            gates = env.gate_indices.detach().cpu().tolist()
            ng = int(getattr(env, "num_gates", 0)) or 1
            gi = int(env.gate_indices[0].item())
            print(
                f"[timer] ep_sim_t={sim_ep_t:.2f}s wall_run={wall_run_t:.2f}s "
                f"gate={gi}/{ng - 1} reward={reward:.4f}"
            )

        track_done = bool(env.track_completed[0].item()) if hasattr(env, "track_completed") else False
        if track_done and not prev_track_done:
            wall_lap = time.perf_counter() - wall_ep0
            print(
                f"[LAP TIME] sim={sim_ep_t:.3f}s wall={wall_lap:.3f}s "
                f"(policy_dt={policy_dt:.4f}s/step)"
            )
        prev_track_done = track_done

        done = next_td["done"].squeeze(-1)
        if done.any():
            reset_td = TensorDict(
                {"_reset": done},
                batch_size=env.batch_size,
                device=env.device,
            )
            td = env.reset(reset_td)
            ep_step = 0
            wall_ep0 = time.perf_counter()
            prev_track_done = bool(env.track_completed[0].item()) if hasattr(env, "track_completed") else False
        else:
            td = next_td

    simulation_app.close()

if __name__ == "__main__":
    main()
