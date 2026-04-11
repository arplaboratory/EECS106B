## Distrobox Setup (EECS106B)

This codebase is based on Distrobox, [Docker](https://docs.docker.com/get-started/docker-overview/) and Podman. We abstract away most of their inner workings, but it is recommended to have a preliminary understanding of each of these packages by looking at their official documentations. 

### Step 1: Prereqs (host)

- On your personal github accounnt, create a **private** GitHub repository named `EECS106B` (do not fork this repo). On [github.com](https://github.com), click "New repository", name it `EECS106B`, set visibility to **Private**, and create it **without** initializing with a README, .gitignore, or license (leave all unchecked). You must create this empty repo on GitHub first before you can push to it.
Then clone this repo, point it at your new private remote, and initialize submodules. The git submodule command can take a while:
  ```
  git clone git@github.com:arplaboratory/EECS106B.git
  cd EECS106B
  git remote set-url origin git@github.com:<your-username>/EECS106B.git
  git push -u origin main
  git submodule update --init --recursive
  ```
  Replace `<your-username>` with your GitHub username. **Keep your repository private.** All development should happen in your private repo.
- Make sure you can authenticate with GitHub (preferably set up SSH keys).
- Add these to your host `~/.bashrc` (or `~/.zshrc`) (replace with your paths):
  ```
  export EECS106B_DIR="/path/to/EECS106B"
  source "$EECS106B_DIR/helpers"
  ```
  Or run this code to append them to `~/.bashrc`:
  ```
  if [[ "$(pwd)" == */EECS106B ]]; then
    cat <<EOF >> ~/.bashrc
  export EECS106B_DIR="$(pwd)"
  source "\$EECS106B_DIR/helpers"
  EOF
  else
    echo "ERROR: run this from your EECS106B repo directory." >&2
    return 1
  fi
  ```
- Setup [wandb](https://docs.wandb.ai/models/quickstart) to visualize training statistics. Once you have your WANDB_API_KEY, add it to the  `.bashrc` file in the `EECS106B` git folder. Alternatively, run the following command
  ```
  cat <<EOF >> "$EECS106B_DIR/.bashrc"


  export WANDB_API_KEY="<your_api_key>"
  EOF
  ```
- Make sure to re-source using `source ~/.bashrc`

### Step 2: Create the distrobox
Luckily, we can re-use the grasping distrobox
Re-enter the distrobox:

```
distrobox enter grasping
```
We set up a virtual environment, that you can source with 

```
source /opt/drone_venv/bin/activate
```

You will also need to source the isaac lab venv as the same in project 4b
```
source /workspace/isaacsim/setup_conda_env.sh
```

### Step 3: Train a hover policy

Let's train a simple drone hover controller using the task defined in `cfg/task/Hover.yaml`. This will train a policy that outputs thrust and bodyrate commands to hover in a static position. The number of envs are specified in the yaml file's `num_envs` argument. The environment observation space, and rewards are defined in `EECS106B/omni_drones/envs/single/hover.py`. Make sure you understand these two files, since you will have to create similar files for a drone racing task.

The `drone_model["controller"]` field in `cfg/task/Hover.yaml`, specifies the controller type and therefore the action space of the policy. `RateController` is a body rate controller, so the drone accepts thrust and body rates commands. Therefore the policy will output 4 values. 

Inside distrobox,

```
python3 scripts/train.py algo=ppo headless=true
```

If you haven't setup wandb, run `python train.py algo=ppo headless=true wandb.mode=disabled` instead. If you see PPO training logs (e.g., average reward metrics), your setup is working.

After training is complete, visualize the results. You should see the drone reach the goal state. 

```
python play.py task.env.num_envs=1 algo.checkpoint_path=</tmp/wandb/run--runid/files/checkpoint_final.pt> headless=False
```

If you haven't configured wandb, the checkpoint files will be saved in  `/tmp/wandb/`.

### Step 4: Racing environment

The racing environment is defined in `envs/drone_race/drone_race.py`. We provide the user with code for extracting the relevant observations. You must design the reward function in `_compute_reward_and_done`. The environment configuration in defined in `cfg/task/DroneRace.yaml` and the ppo parameters are in `cfg/algo/DroneRace.yaml`. Feel free to change any other parts of the pipeline, this is simply a good starting point. 

**What you should not change:** Drone dynamics including the drone's physical parameters and constraints. Gate design and gate locations.

### How to stop and remove the container

You can run `exit` inside a distrobox container to exit the container. This will bring you back into the host system's bash.

If you'd like to end the container (in case you want to reset the container, etc), run 

```
distrobox_end
```

from the host environment to stop and remove the distrobox container.
