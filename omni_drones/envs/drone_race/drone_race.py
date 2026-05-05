# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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

import numpy as np
import imp
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec, BinaryDiscreteTensorSpec

import isaacsim.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import (
    euler_to_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quat_axis,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
    normalize,
)
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

from pxr import UsdPhysics

from .drone_planner import DronePlanner, PlannerParams

# Debug visualization
try:
    from isaacsim.util.debug_draw import _debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    _debug_draw = None

class DroneRaceEnv(IsaacEnv):
    r"""
    A drone racing task where the agent must navigate through a sequence of gates
    in a racing track. The gates are arranged in a track pattern and the agent
    must pass through them in order.

    ## Observation

    - `drone_state` (15): Custom state vector `[lin_vel(3) | rot_mat_flat(9) | ang_vel(3)]`. Note that
      position is **not** included.
    - `next_gate_rpos` (3): The relative position of the next gate to the drone in the drone's local frame.
    - `next_to_next_gate_pos` (3): The position of the gate after the immediate next gate, expressed in
      the next gate's local frame. Clamped at the last gate (no wrap-around).
    - `next_gate_rot_mat_2col` (6): The first two columns of the next gate's rotation matrix in the world frame
      (i.e. the gate's local x- and y-axes expressed in world coordinates), flattened to a 6-vector.

    ## Reward  *(student implementation required)*

    **Your task is to implement the reward function in `_compute_reward_and_done`.**

    ## Episode End

    The episode ends when the drone crashes (physical contact or off-course),
    completes the full lap, or the maximum episode length is reached.
    **You should also implement the crash/termination condition** in
    `_compute_reward_and_done` (currently returns all-zeros as a placeholder).

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `track_config`          | dict  | None          | Optional dictionary defining gate positions and orientations. If provided, gates are placed according to this config. Format: `{"1": {"pos": (x, y, z), "yaw": angle}, ...}`. If None, uses circular track.                            |
    | `num_gates`             | int   | 8             | Number of gates in the racing track (only used if `track_config` is None).                                                                                                                                                              |
    | `track_radius`          | float | 5.0           | Radius of the circular track (only used if `track_config` is None).                                                                                                                                                                     |
    | `gate_spacing`          | float | 3.0           | Spacing between gates along the track (only used if `track_config` is None).                                                                                                                                                           |
    | `gate_height`           | float | 2.0           | Height of the gates (only used if `track_config` is None).                                                                                                                                                                              |
    | `gate_scale`            | float | 1.0           | Scale of the gate assets.                                                                                                                                                                                                              |
    | `gate_asset_path`       | str   | None          | Path to the gate USD asset. Defaults to `ASSET_PATH/gate/gate.usd` (isaac_drone_racer style). Can be overridden in config.                                                                                                        |
    """
    def __init__(self, cfg, headless):
        # -----------------------------------------------------------------------
        # STUDENT TODO (1/3): Load your reward scaling hyperparameters.
        #
        # Each line reads one value from cfg/task/DroneRace.yaml.
        # The second argument to .get() is the default used if the key is absent.
        # Add matching entries in cfg/task/DroneRace.yaml for each one you add.
        #
        # Example:
        #   self.reward_progress_scale = cfg.task.get("reward_progress_scale", 1.0)
        #   self.reward_gate_passage   = cfg.task.get("reward_gate_passage", 10.0)
        #   self.reward_crash_scale    = cfg.task.get("reward_crash_scale", 5.0)
        # -----------------------------------------------------------------------
        # ----- ADD YOUR REWARD CONFIG LINES BELOW (replace / extend the example) -----

        # self.reward_example = cfg.task.get("reward_example", 0.5)

        # ----- END STUDENT CODE -----
        self.trajectory_cache = None
        self.trajectory_mode = str(cfg.task.get("trajectory_mode", "planner")).lower()
        self.trajectory_forward_speed = cfg.task.get("trajectory_forward_speed", 1.0)
        self.trajectory_spin_yaw_rate = cfg.task.get("trajectory_spin_yaw_rate", 1.0)
        self.trajectory_circle_radius = cfg.task.get("trajectory_circle_radius", 1.5)
        self.trajectory_circle_speed = cfg.task.get("trajectory_circle_speed", 1.0)
        self.gate_scale = cfg.task.gate_scale
        
        # Gate asset path - default to isaac_drone_racer gate asset
        # User can override this in config: gate_asset_path: "path/to/gate.usd"
        # If not specified, defaults to gate/gate.usd (isaac_drone_racer style)
        # Gate asset is in ASSET_PATH/gate/gate.usd
        self.gate_asset_path = cfg.task.get("gate_asset_path", ASSET_PATH + "/gate/gate_a2rl.usd")
        
        # Track configuration: support both config-based and circular track
        self.track_config = cfg.task.get("track_config", None)
        if self.track_config is not None:
            # Config-based track: gates defined with positions and yaw angles
            self.num_gates = len(self.track_config)
            self.track_type = "config"
            # For config-based tracks, use gate_height from config or default
            self.gate_height = cfg.task.get("gate_height", 2.0)
        else:
            # Circular track (backward compatibility)
            self.num_gates = int(cfg.task.num_gates)
            self.track_radius = cfg.task.track_radius
            self.gate_spacing = cfg.task.gate_spacing
            self.gate_height = cfg.task.gate_height
            self.track_type = "circular"
        
        import traceback
        import sys
        
        print(f"[DroneRaceEnv] Initializing, num_gates={self.num_gates}")
        try:
            super().__init__(cfg, headless)
            print(f"[DroneRaceEnv] super().__init__ completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed in super().__init__")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        try:
            self.drone.initialize(track_contact_forces=True)
            print(f"[DroneRaceEnv] drone.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print("ERROR: Failed to initialize drone")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)
            raise

        self.hover_cmd_thrust = None
        self.policy_dt = self.dt * self.substeps
        self.local_planner = DronePlanner(PlannerParams(dt=float(self.policy_dt)))
        self._mpc_u_prev = np.zeros(4, dtype=np.float64)
        self.test_traj_origin = torch.zeros(self.num_envs, 3, device=self.device)
        self.test_traj_forward_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.test_traj_left_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self.test_traj_yaw0 = torch.zeros(self.num_envs, 1, device=self.device)
        self.trajectory_cache = [None for _ in range(self.num_envs)]

        print(f"[DroneRaceEnv] num_envs={self.num_envs}, num_gates={self.num_gates}")
        # Track gate progress for each environment
        self.gate_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gate_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.track_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.prev_distance_to_gate = torch.zeros(self.num_envs, device=self.device)
        # Gate crossing detection: drone position in gate frame from previous step
        self.gate_width = cfg.task.get("gate_width", 1.0)
        self.prev_drone_in_gate_frame = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_action = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device)
        self.effort = torch.zeros(self.num_envs, 1, self.drone.action_spec.shape[-1], device=self.device)
        # For MPC w/ local planner
        self.local_planner.params.gate_width = float(self.gate_width)
        self.local_planner.params.gate_height = float(self.gate_height)
        self.local_planner.params.mass = float(self.drone.MASS_0.reshape(-1)[0].item())
        J0 = self.drone.INERTIA_0.reshape(-1)
        self.local_planner.params.Jx, self.local_planner.params.Jy, self.local_planner.params.Jz = [float(J0[i].item()) for i in range(3)]
        if self.controller is not None and hasattr(self.controller, "max_body_rates"):
            self.local_planner.params.max_body_rate = float(self.controller.max_body_rates.max().item())

        # Use a single view with wildcard pattern to access all gates
        try:
            print(f"[DroneRaceEnv] Creating RigidPrimView with pattern='/World/envs/env_*/Gate_*', shape=[{self.num_envs}, {self.num_gates}]")
            self.gates = RigidPrimView(
                "/World/envs/env_*/Gate_*",
                reset_xform_properties=False,
                shape=[self.num_envs, self.num_gates],
                track_contact_forces=False
            )
            print(f"[DroneRaceEnv] RigidPrimView created, calling initialize()...")
            self.gates.initialize()
            print(f"[DroneRaceEnv] gates.initialize() completed")
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed to initialize gates view with num_envs={self.num_envs}, num_gates={self.num_gates}")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed to initialize gates view\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise  # Re-raise to see the full error
        
        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 1.5], device=self.device),
            torch.tensor([1.0, 1.0, 2.5], device=self.device)
        )
        self.init_rp_dist = D.Uniform(
            torch.tensor([-.2, -.2], device=self.device) * torch.pi,
            torch.tensor([.2, .2], device=self.device) * torch.pi
        )

        self.offset_local = torch.tensor([-1.5, 0.0, self.gate_height / 2.0], device=self.device)
        self.alpha = 0.8
        
        # Debug visualization: enable via config (default: False)
        self.debug_gate_origins = cfg.task.get("debug_gate_origins", False)
        if self.debug_gate_origins and DEBUG_DRAW_AVAILABLE:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            self.axis_length = cfg.task.get("debug_axis_length", 0.3)  # Length of axis lines
        else:
            self.draw = None

    def _draw_gate_origins(self, gate_world_pos, gate_world_rot, env_idx=0):
        """
        Draw coordinate axes at gate positions to visualize where the gate frame origin is.
        
        Args:
            gate_world_pos: (num_envs, num_gates, 3) tensor of gate positions in world coordinates
            gate_world_rot: (num_envs, num_gates, 4) tensor of gate rotations (quaternions) in world coordinates
            env_idx: Which environment to visualize (default: 0, first environment)
        """
        if self.draw is None:
            return
        
        # Clear previous lines
        self.draw.clear_lines()
        
        # Select one environment to visualize
        gate_pos = gate_world_pos[env_idx]  # (num_gates, 3) - world coordinates
        gate_rot = gate_world_rot[env_idx]  # (num_gates, 4) - world coordinates
        
        # Define axis directions in local frame (gate's local frame)
        # X-axis (red): forward direction
        # Y-axis (green): right direction  
        # Z-axis (blue): up direction
        axis_dirs_local = torch.tensor([
            [self.axis_length, 0.0, 0.0],  # X-axis (red)
            [0.0, self.axis_length, 0.0],  # Y-axis (green)
            [0.0, 0.0, self.axis_length],  # Z-axis (blue)
        ], device=self.device, dtype=torch.float32)  # (3, 3)
        
        # Colors: Red for X, Green for Y, Blue for Z (RGBA)
        axis_colors = [
            (1.0, 0.0, 0.0, 1.0),  # Red for X
            (0.0, 1.0, 0.0, 1.0),  # Green for Y
            (0.0, 0.0, 1.0, 1.0),  # Blue for Z
        ]
        
        # Draw axes for each gate
        for gate_idx in range(gate_pos.shape[0]):
            gate_origin = gate_pos[gate_idx]  # (3,)
            gate_quat = gate_rot[gate_idx]  # (4,)
            
            # Rotate axis directions from gate's local frame to world frame
            # quat_rotate expects (N, 4) and (N, 3), so we need to rotate each axis separately
            # Expand gate_quat to match number of axes (3 axes)
            gate_quat_expanded = gate_quat.unsqueeze(0).expand(3, -1)  # (3, 4)
            axis_dirs_world = quat_rotate(
                gate_quat_expanded,  # (3, 4)
                axis_dirs_local  # (3, 3) - each row is a 3D vector
            )  # (3, 3) - each row is a rotated 3D vector
            
            # Draw each axis
            for axis_idx, (axis_dir, color) in enumerate(zip(axis_dirs_world, axis_colors)):
                start_point = gate_origin.cpu().tolist()
                end_point = (gate_origin + axis_dir).cpu().tolist()
                
                # Draw line from origin to end point
                self.draw.draw_lines(
                    [start_point],
                    [end_point],
                    [color],
                    [3.0]  # Line width
                )

    def _design_scene(self):
        print("Designing scene")
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Create gates based on track configuration
        scale = torch.ones(3) * self.gate_scale
        gate_positions_list = []
        gate_orientations_list = []
        
        # Config-based track: gates defined with positions and yaw angles
        # Sort gate keys to ensure correct order
        gate_keys = sorted(self.track_config.keys(), key=lambda x: int(x))
        
        for i, gate_key in enumerate(gate_keys):
            gate_cfg = self.track_config[gate_key]
            pos = gate_cfg.get("pos", (0.0, 0.0, 1.0))
            yaw = gate_cfg.get("yaw", 0.0)
            
            # Convert to torch tensors
            if isinstance(pos, (list, tuple)):
                gate_pos = torch.tensor(pos, device=self.device, dtype=torch.float32)
            else:
                gate_pos = torch.tensor([pos[0], pos[1], pos[2]], device=self.device, dtype=torch.float32)
            
            if isinstance(yaw, torch.Tensor):
                gate_yaw = yaw.item() if yaw.numel() == 1 else yaw
            else:
                gate_yaw = float(yaw)
            
            # Create quaternion from yaw (rotation around z-axis)
            gate_orientation = euler_to_quaternion(
                torch.tensor([0., 0., gate_yaw], device=self.device)
            )
            
            gate_positions_list.append(gate_pos)
            gate_orientations_list.append(gate_orientation)
            
            # Spawn gate using configured gate asset
            gate_prim = prim_utils.create_prim(
                f"/World/envs/env_0/Gate_{i}",
                usd_path=self.gate_asset_path,
                translation=(gate_pos[0].item(), gate_pos[1].item(), gate_pos[2].item()),
                orientation=(gate_orientation[0].item(), gate_orientation[1].item(), 
                                gate_orientation[2].item(), gate_orientation[3].item()),
                scale=scale
            )
            # Make gate static: disable gravity and make kinematic
            gate_prim_path = f"/World/envs/env_0/Gate_{i}"
            kit_utils.set_nested_rigid_body_properties(
                gate_prim_path,
                disable_gravity=True,
                linear_damping=1000.0,  # Very high damping to prevent movement
                angular_damping=1000.0,
            )
            # Set kinematic on all nested rigid bodies to prevent movement from collisions
            gate_prim_obj = prim_utils.get_prim_at_path(gate_prim_path)
            all_prims = [gate_prim_obj]
            while len(all_prims) > 0:
                child_prim = all_prims.pop(0)
                if child_prim.HasAttribute("physics:kinematicEnabled"):
                    child_prim.GetAttribute("physics:kinematicEnabled").Set(True)
                all_prims += child_prim.GetChildren()

        # Gate positions and orientations will be retrieved from views at runtime
        # No need to store them manually since gates are static

        # Spawn drone at start position (behind first gate, in gate's local frame)
        # The gate's local x-axis points in the tangent direction (for circular) or forward (for config)
        # We want to spawn the drone behind the gate, so we offset backward along the gate's x-axis
        first_gate_pos = gate_positions_list[0]
        first_gate_rot = gate_orientations_list[0]
        # Offset backward in gate's local frame (negative x direction)
        offset_local = torch.tensor([-1.5, 0.0, self.gate_height / 2.0], device=self.device)
        # Rotate offset to world frame - quat_rotate expects batched inputs, so add batch dimension
        offset_world = quat_rotate(first_gate_rot.unsqueeze(0), offset_local.unsqueeze(0)).squeeze(0)
        start_pos = first_gate_pos + offset_world
        # Store first gate position as landing target after completing all laps
        self.first_gate_pos = first_gate_pos
        self.first_gate_rot = first_gate_rot
        self.drone.spawn(translations=[(start_pos[0].item(), start_pos[1].item(), start_pos[2].item())])


        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # Custom robot state: linear_vel(3) + rotation_matrix_flat(9) + angular_vel(3) = 18
        robot_state_dim = 3 + 9 + 3  # 15
        # Observation: robot_state(15) + next_gate_rpos_local(3) + next_to_next_gate_pos(3)
        #              + next_gate_rot_mat_2col(6)
        observation_dim = robot_state_dim + 3 + 3 + 6
        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            },
            "info": {
                "drone_state": Unbounded((1, robot_state_dim), device=self.device),
                "drone_pos": Unbounded((1, 3), device=self.device),
                "drone_rot": Unbounded((1, 4), device=self.device),
                "drone_vel": Unbounded((1, 3), device=self.device),
                "drone_ang_vel_body": Unbounded((1, 3), device=self.device),
                "current_gate_center": Unbounded((1, 3), device=self.device),
                "current_gate_rot": Unbounded((1, 4), device=self.device),
                "next_gate_center": Unbounded((1, 3), device=self.device),
                "gate_forward_axis": Unbounded((1, 3), device=self.device),
                "next_segment": Unbounded((1, 3), device=self.device),
                "drone_in_gate_frame": Unbounded((1, 3), device=self.device),
            },
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )
        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "gates_passed": Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "collision": Unbounded(1),
            "crashed_z": Unbounded(1),
            "crashed_z_gate": Unbounded(1),
            "crashed_distance": Unbounded(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
            "truncated":  Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        
        # Reset gate progress
        self.gate_indices[env_ids] = 0
        self.gate_passed[env_ids] = False
        self.track_completed[env_ids] = False
        self.last_action[env_ids] = 0.0
        self.effort[env_ids] = 0.0 # Adding this line changes the result
        if self.trajectory_cache is not None:
            for env_id in env_ids.tolist():
                self.trajectory_cache[int(env_id)] = None
        self._mpc_u_prev[:] = 0.0

        # Reset gate velocities to prevent drift (gates are static, so we just zero velocities)
        # Set velocities to zero for all gates in reset environments
        num_gates_to_reset = len(env_ids) * self.num_gates
        gate_velocities = torch.zeros(num_gates_to_reset, 6, device=self.device)
        
        # Reshape to match gate view shape: (num_envs, num_gates, 6)
        gate_velocities = gate_velocities.reshape(len(env_ids), self.num_gates, 6)
        self.gates.set_velocities(gate_velocities, env_indices=env_ids)

        # Reset drone position and orientation
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        try:
        
            # Position drone near the first gate
            # Get gate positions from views - get all gates first, then select the ones we need
            # This avoids the unflatten issue when using env_indices
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (num_envs, num_gates, 3), (num_envs, num_gates, 4)
            # Select only the environments we're resetting
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
            gate_env_pos = gate_env_pos[env_ids]  # (len(env_ids), num_gates, 3)
            gate_env_rot = gate_env_rot[env_ids]  # (len(env_ids), num_gates, 4)
            first_gate_pos = gate_env_pos[:, 0]  # (N, 3)
            first_gate_rot = gate_env_rot[:, 0]  # (N, 4)
            
            # Calculate offset in gate's local frame (behind the gate)
            # Gate's local x-axis points in the forward direction 
            
            # Rotate offset to world frame using gate's orientation
            # Expand offset_local to match batch size
            offset_local_expanded = self.offset_local.unsqueeze(0).expand(len(env_ids), -1)  # (N, 3)
            offset_world = quat_rotate(first_gate_rot, offset_local_expanded)  # (len(env_ids), 3)

            # TODO You can comment out lines below for random drone start position
            # pos_perturbation = self.init_pos_dist.sample(env_ids.shape) - self.init_pos_dist.mean
            # drone_start_pos = first_gate_pos + offset_world + pos_perturbation  # (len(env_ids), 3)
            drone_start_pos = first_gate_pos + offset_world  # (len(env_ids), 3)
            
            drone_start_pos_with_agent = drone_start_pos.unsqueeze(1)  # (len(env_ids), 1, 3)
            env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)

            self.drone.set_world_poses(
                drone_start_pos_with_agent + env_positions_with_agent,
                drone_rot, env_ids
            )
        except Exception as e:
            import traceback
            print("=" * 80)
            print(f"ERROR: Failed in _reset_idx (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            raise RuntimeError(
                f"DroneRaceEnv._reset_idx failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids
        )

        self.drone.set_joint_positions(torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids)
        self.drone.set_joint_velocities(torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids)

        gate_center_offset_local = torch.tensor(
            [0.0, 0.0, self.gate_height / 2.0],
            device=self.device
        )
        gate_center_offset_local_expanded = gate_center_offset_local.unsqueeze(0).expand(len(env_ids), -1)
        gate_center_offset_world = quat_rotate(first_gate_rot, gate_center_offset_local_expanded)
        first_gate_center = first_gate_pos + gate_center_offset_world
        self.prev_distance_to_gate[env_ids] = torch.norm(
            first_gate_center - drone_start_pos,
            dim=-1
        )

        # Initialise drone position in gate frame for crossing detection
        drone_to_first_gate_center = drone_start_pos - first_gate_center  # (len(env_ids), 3)
        self.prev_drone_in_gate_frame[env_ids] = quat_rotate_inverse(
            first_gate_rot, drone_to_first_gate_center
        )  # (len(env_ids), 3)

        gate_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        gate_forward_local = gate_forward_local.unsqueeze(0).expand(len(env_ids), -1)
        test_forward_dir = quat_rotate(first_gate_rot, gate_forward_local)
        test_forward_dir[..., 2] = 0.0
        test_forward_dir = normalize(test_forward_dir)
        test_left_dir = torch.stack(
            [
                -test_forward_dir[:, 1],
                test_forward_dir[:, 0],
                torch.zeros_like(test_forward_dir[:, 0]),
            ],
            dim=-1,
        )
        self.test_traj_origin[env_ids] = drone_start_pos
        self.test_traj_forward_dir[env_ids] = test_forward_dir
        self.test_traj_left_dir[env_ids] = test_left_dir
        self.test_traj_yaw0[env_ids] = yaw.squeeze(1)

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        '''
        Input actions are in scaled units 
        '''
        actions = tensordict[("agents", "action")].clone()
        if self.controller is not None:
            root_state = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds = self.controller(root_state, *raw_actions)            
            _ = self.drone.apply_action(rotor_cmds)
        else:
            raise Exception("No controller found. This is not yet supported.")

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.effort = tensordict[("agents", "action")].clone()

    def _build_robot_state(self) -> torch.Tensor:
        """Builds the custom robot state vector used for observations.

        Calls ``drone.get_state()`` to refresh all cached kinematics, then
        concatenates along the feature dimension:

            [position (3) | linear_velocity (3) | rotation_matrix_flat (9) | angular_velocity (3)]

        Returns:
            Tensor of shape (N, 1, 18).
        """
        self.drone.get_state()  # refresh pos, rot, vel_w, vel_b caches
        lin_vel = self.drone.get_linear_velocity()   # (N, 1, 3)
        rot_mat = self.drone.get_rotation_matrix()   # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()  # (N, 1, 3)
        return torch.cat([lin_vel, rot_mat, ang_vel], dim=-1)  # (N, 1, 18)
        
    def get_relative_gate_position(self, gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot):
        """Return gate position/rotation relative to each drone, both in world and drone-local frames.

        Args:
            gate_indices: (N,) integer tensor – the target gate index per environment.
            gate_env_pos: (N, num_gates, 3) gate positions in env frame.
            gate_env_rot: (N, num_gates, 4) gate rotations in env frame.
            drone_pos:    (N, 1, 3) drone positions.
            drone_rot:    (N, 1, 4) drone rotations.

        Returns:
            next_gate_pos:        (N, 1, 3) selected gate position in env frame.
            next_gate_rot:        (N, 1, 4) selected gate rotation in env frame.
            next_gate_rpos_world: (N, 1, 3) gate position relative to drone, in world frame.
            next_gate_rpos_local: (N, 1, 3) gate position relative to drone, in drone-local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        next_gate_pos = gate_env_pos[batch_indices, gate_indices]  # (N, 3)
        next_gate_rot = gate_env_rot[batch_indices, gate_indices]  # (N, 4)

        # Expand to match agent dimension for broadcasting
        next_gate_pos = next_gate_pos.unsqueeze(1)  # (N, 1, 3)
        next_gate_rot = next_gate_rot.unsqueeze(1)  # (N, 1, 4)

        # Relative position in world frame
        next_gate_rpos_world = next_gate_pos - drone_pos  # (N, 1, 3)

        # Relative position in drone-local frame
        drone_rot_flat = drone_rot.squeeze(1)                          # (N, 4)
        next_gate_rpos_world_flat = next_gate_rpos_world.squeeze(1)    # (N, 3)
        next_gate_rpos_local_flat = quat_rotate_inverse(drone_rot_flat, next_gate_rpos_world_flat)  # (N, 3)
        next_gate_rpos_local = next_gate_rpos_local_flat.unsqueeze(1)  # (N, 1, 3)

        return next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local

    def get_next_to_next_gate_position(self, next_to_next_gate_indices, gate_env_pos, gate_env_rot, next_gate_indices):
        """Return the position of the next-to-next gate expressed in the next gate's local frame.

        Args:
            next_to_next_gate_indices: (N,) index of the gate after the immediate next gate (clamped at last gate).
            gate_env_pos:              (N, num_gates, 3) gate positions in env frame.
            gate_env_rot:              (N, num_gates, 4) gate rotations in env frame.
            next_gate_indices:         (N,) index of the immediate next gate per environment.

        Returns:
            (N, 3) position of the next-to-next gate relative to the next gate, in the next gate's local frame.
        """
        batch_indices = torch.arange(self.num_envs, device=self.device)

        # Positions/orientations of the immediate next gate in env frame
        next_gate_pos_env = gate_env_pos[batch_indices, next_gate_indices]      # (N, 3)
        next_gate_rot_env = gate_env_rot[batch_indices, next_gate_indices]      # (N, 4)

        # Position of the next-to-next gate in env frame
        n2n_gate_pos_env = gate_env_pos[batch_indices, next_to_next_gate_indices]  # (N, 3)

        # Relative position in env frame
        rpos_env = n2n_gate_pos_env - next_gate_pos_env  # (N, 3)

        # Rotate into the next gate's local frame
        rpos_next_gate_frame = quat_rotate_inverse(next_gate_rot_env, rpos_env)  # (N, 3)

        return rpos_next_gate_frame


    def _compute_state_and_obs(self):
        import traceback
        import sys
        
        try:
            # Build custom state: [pos(3) | lin_vel(3) | rot_mat_flat(9) | ang_vel(3)] -> (N, 1, 18)
            # Also refreshes drone.pos / drone.rot caches used below for gate computations.
            self.drone_state = self._build_robot_state()
            drone_pos = self.drone.pos   # (N, 1, 3) refreshed by _build_robot_state
            drone_rot = self.drone.rot   # (N, 1, 4) refreshed by _build_robot_state

            # Get gate positions from views (similar to fly_through.py)
            # gates.get_world_poses() returns (pos, rot) with shape (num_envs, num_gates, ...)
            gate_world_pos, gate_world_rot = self.gates.get_world_poses()  # (N, num_gates, 3), (N, num_gates, 4)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_state_and_obs (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            sys.stderr.write("=" * 80 + "\n")
            sys.stderr.write(f"ERROR: Failed in _compute_state_and_obs\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 80 + "\n")
            raise RuntimeError(
                f"DroneRaceEnv._compute_state_and_obs failed (num_envs={self.num_envs}, num_gates={self.num_gates})"
            ) from e

        gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))  # (N, num_gates, 3), (N, num_gates, 4)
        
        # If track is completed, target first gate for landing
        track_completed = self.track_completed  # (N,)
        
        # Get current gate positions for each environment
        target_gate_indices = self.gate_indices  # (N,)

        next_gate_pos, next_gate_rot, next_gate_rpos_world, next_gate_rpos_local = (
            self.get_relative_gate_position(
                target_gate_indices, gate_env_pos, gate_env_rot, drone_pos, drone_rot
            )
        )
        
        # Get next-to-next gate positions. If the immediate next gate is the last gate, clamp so
        # next-to-next points at the same gate (no wrap-around).
        next_to_next_gate_indices = torch.where(
            target_gate_indices == self.num_gates - 1,
            target_gate_indices,
            target_gate_indices + 1,
        )  # (N,)

        next_to_next_gate_pos = self.get_next_to_next_gate_position(
            next_to_next_gate_indices, gate_env_pos, gate_env_rot, target_gate_indices
        )  # (N, 3)

        # Angle between drone->gate vector and gate normal (gate local +x axis in env frame)
        gate_normal_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        gate_normal_local_expanded = gate_normal_local.unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        gate_normal_world = quat_rotate(next_gate_rot.squeeze(1), gate_normal_local_expanded)  # (N, 3)
        gate_vec_world = next_gate_rpos_world.squeeze(1)  # (N, 3)
        gate_vec_norm = torch.norm(gate_vec_world, dim=-1).clamp_min(1e-6)
        gate_normal_norm = torch.norm(gate_normal_world, dim=-1).clamp_min(1e-6)
        cos_angle = (gate_vec_world * gate_normal_world).sum(-1) / (gate_vec_norm * gate_normal_norm)
        gate_angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # (N,)

        # Gate progress: fraction of gates completed in the single lap
        gate_progress = self.gate_indices.float() / self.num_gates  # (N,)
        gate_progress = torch.where(track_completed, torch.ones_like(gate_progress), gate_progress)
        
        # Next gate orientation: first 2 columns of its rotation matrix in the world frame.
        # Rotate the gate-local x- and y-axes into world coordinates via quat_rotate,
        # then concatenate to a 6-vector per environment.
        next_gate_rot_flat = next_gate_rot.squeeze(1)  # (N, 4)
        e_x = torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        e_y = torch.tensor([0., 1., 0.], device=self.device).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        gate_col0 = quat_rotate(next_gate_rot_flat, e_x)  # (N, 3) - gate x-axis in world frame
        gate_col1 = quat_rotate(next_gate_rot_flat, e_y)  # (N, 3) - gate y-axis in world frame
        next_gate_rot_mat_2col = torch.cat([gate_col0, gate_col1], dim=-1).unsqueeze(1)  # (N, 1, 6)

        # Build observation
        # All components need to have the agent dimension (middle dimension) to match spec (N, 1, obs_dim)
        obs = [
            self.drone_state,  # (N, 1, state_dim) - already has agent dimension
            next_gate_rpos_local,  # (N, 1, 3) - already has agent dimension
            next_to_next_gate_pos.unsqueeze(1),  # (N, 1, 3)
            next_gate_rot_mat_2col,  # (N, 1, 6)
        ]
        
        # Concatenate along last dimension: (N, 1, obs_dim)
        obs = torch.cat(obs, dim=-1)  # (N, 1, obs_dim)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "info": {
                    "drone_state": self.drone_state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def plan_test_trajectory(self):
        """Return a simple trajectory to test the geometric controller."""
        self.drone.get_state()
        drone_pos = self.drone.pos.squeeze(1)
        t = (self.progress_buf * self.policy_dt).unsqueeze(-1)
        x_des = self.test_traj_origin - drone_pos
        v_des = torch.zeros(self.num_envs, 3, device=self.device)
        a_des = torch.zeros(self.num_envs, 3, device=self.device)
        yaw_des = self.test_traj_yaw0.clone()
        yaw_rate_des = torch.zeros(self.num_envs, 1, device=self.device)

        if self.trajectory_mode == "hover":
            pass
        elif self.trajectory_mode in ("straight", "go_straight", "forward"):
            ref_pos = self.test_traj_origin + (
                self.trajectory_forward_speed * t
            ) * self.test_traj_forward_dir
            x_des = ref_pos - drone_pos
            v_des = self.trajectory_forward_speed * self.test_traj_forward_dir
            yaw_des = torch.atan2(
                self.test_traj_forward_dir[:, 1],
                self.test_traj_forward_dir[:, 0],
            ).unsqueeze(-1)
        elif self.trajectory_mode == "spin":
            yaw_des = self.test_traj_yaw0 + self.trajectory_spin_yaw_rate * t
            yaw_rate_des = torch.full(
                (self.num_envs, 1),
                self.trajectory_spin_yaw_rate,
                device=self.device,
            )
        elif self.trajectory_mode == "circle":
            omega = self.trajectory_circle_speed / max(
                self.trajectory_circle_radius, 1e-5
            )
            phase = omega * t
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            circle_offset = (
                self.trajectory_circle_radius * (cos_phase - 1.0)
            ) * self.test_traj_forward_dir + (
                self.trajectory_circle_radius * sin_phase
            ) * self.test_traj_left_dir
            ref_pos = self.test_traj_origin + circle_offset
            x_des = ref_pos - drone_pos
            v_des = (
                -self.trajectory_circle_radius * omega * sin_phase
            ) * self.test_traj_forward_dir + (
                self.trajectory_circle_radius * omega * cos_phase
            ) * self.test_traj_left_dir
            a_des = (
                -self.trajectory_circle_radius * omega * omega * cos_phase
            ) * self.test_traj_forward_dir + (
                -self.trajectory_circle_radius * omega * omega * sin_phase
            ) * self.test_traj_left_dir
        else:
            raise ValueError(
                f"Unknown trajectory_mode '{self.trajectory_mode}'. "
                "Supported modes: planner, hover, straight, spin, circle."
            )

        return {
            "x_des": x_des,
            "v_des": v_des,
            "a_des": a_des,
            "yaw_des": yaw_des,
            "yaw_rate_des": yaw_rate_des,
        }

    def _consume_cached_trajectory_step(self, env_idx, gate_index, drone_pos):
        if self.trajectory_cache is None:
            return None

        cache = self.trajectory_cache[env_idx]
        if cache is None or cache["gate_index"] != gate_index:
            self.trajectory_cache[env_idx] = None
            return None

        if cache["p_des"].shape[0] == 0:
            self.trajectory_cache[env_idx] = None
            return None

        step = {
            "x_des": cache["p_des"][0] - drone_pos,
            "v_des": cache["v_des"][0],
            "a_des": cache["a_des"][0],
            "yaw_des": cache["yaw_des"][0],
            "yaw_rate_des": cache["yaw_rate_des"][0],
        }

        if cache["p_des"].shape[0] == 1:
            self.trajectory_cache[env_idx] = None
        else:
            for key in ("p_des", "v_des", "a_des", "yaw_des", "yaw_rate_des"):
                cache[key] = cache[key][1:]

        return step

    def _store_trajectory_cache(self, env_idx, gate_index, result):
        if self.trajectory_cache is None:
            self.trajectory_cache = [None for _ in range(self.num_envs)]

        if result.p_des.size == 0:
            self.trajectory_cache[env_idx] = None
            return

        self.trajectory_cache[env_idx] = {
            "gate_index": gate_index,
            "p_des": torch.as_tensor(
                result.p_des, device=self.device, dtype=torch.float32
            ),
            "v_des": torch.as_tensor(
                result.v_des, device=self.device, dtype=torch.float32
            ),
            "a_des": torch.as_tensor(
                result.a_des, device=self.device, dtype=torch.float32
            ),
            "yaw_des": torch.as_tensor(
                result.yaw_des, device=self.device, dtype=torch.float32
            ).unsqueeze(-1),
            "yaw_rate_des": torch.as_tensor(
                result.yaw_rate_des, device=self.device, dtype=torch.float32
            ).unsqueeze(-1),
        }

    def plan_local_trajectory(self):
        import traceback
        import sys

        track_completed = self.track_completed  # (N,)

        try:
            # Use cached pose tensors directly; self.drone_state does not include position/quaternion.
            drone_pos = self.drone.pos  # (N, 1, 3), env frame
            drone_rot = self.drone.rot  # (N, 1, 4), quaternion

            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))

            # Print debug positions in the same frame (env frame) for easier sanity checks.
            gate_idx0 = int(self.gate_indices[0].item())
            # print(f"Drone position (env frame): {torch.round(drone_pos[0, 0] * 100) / 100}")
            if self.debug_gate_origins:
                self._draw_gate_origins(gate_world_pos, gate_world_rot, env_idx=0)

            batch_indices = torch.arange(self.num_envs, device=self.device)
            current_gate_pos = gate_env_pos[batch_indices, self.gate_indices]  # (N, 3)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_reward_and_done (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            traceback.print_exc()
            raise

        current_gate_rot = gate_env_rot[batch_indices, self.gate_indices]  # (N, 4)
        current_gate_center = self._get_gate_center(current_gate_pos, current_gate_rot)  # (N, 3)

        drone_pos_flat = drone_pos.squeeze(1)  # (N, 3)
        distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)  # (N,)

        # --- gate crossing detection ---
        # You either _deteect_gate_crossings or _detect_gate_crossings_via_segments
        # This function call updates the gate indexes
        gate_passed_this_step, gate_index_changed, new_gate_center = self._detect_gate_crossings(
            drone_pos_flat, current_gate_center, current_gate_rot,
            gate_env_pos, gate_env_rot, batch_indices,
        )

        # -----------------------------------------------------------------------
        # STUDENT TODO (MPC 3/3): Implement your reward function.
        #
        # Variables available in this scope:
        #   drone_pos_flat        (N, 3)    drone position in env frame
        #   drone_rot             (N, 1, 4) drone orientation quaternion (w, x, y, z)
        #   distance_to_gate      (N,)      Euclidean distance to current gate centre
        #   current_gate_center   (N, 3)    3-D centre of the current target gate
        #   gate_passed_this_step (N,)      bool – True when drone just passed a gate
        #   gate_index_changed    (N,)      bool – True when target gate index advanced
        #   new_gate_center       (N, 3)    centre of the (possibly new) target gate
        #   crashed_collision     (N,)      bool – True when contact forces are detected
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in body frame
        #   self.gate_indices     (N,)      index of current target gate (0…num_gates-1)
        #   self.track_completed  (N,)      bool – True when the full lap is done
        #
        # Useful helpers (already imported at top of file):
        #   quat_axis(q, axis)          extract body axis vector (0=x, 1=y, 2=z)
        #   quat_rotate(q, v)           rotate vector v by quaternion q
        #   quat_rotate_inverse(q, v)   rotate vector v by inverse of quaternion q
        # -----------------------------------------------------------------------
        # ----- PREPARE YOUR ARGUMENTS FOR YOUR LOCAL TRAJECTORY PLANNER BELOW (change args)-----
        args = 0
        result = self.local_planner.solve(args)
        # ----- OR COMPUTE DIRECTLY -----

        return {
            "x_des": result["x_des"],
            "v_des": result["v_des"],
            "a_des": result["a_des"],
            "yaw_des": result["yaw_des"],
            "yaw_rate_des": result["yaw_rate_des"],
        }

    def geometric_controller(self, x_des, v_des, yaw_des, a_des=None, yaw_rate_des=None):
        drone_pos = self.drone.pos  # (N, 1, 3), env frame
        drone_rot = self.drone.rot  # (N, 1, 4), quaternion
        R = quaternion_to_rotation_matrix(drone_rot) # (N, 3, 3), rotation matrix
        drone_pos_flat = drone_pos.squeeze(1)  # (N, 3)
        distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)  # (N,)

        kp = torch.tensor([9.6, 9.6, 16.0], device=self.device)
        kv = torch.tensor([4.8, 4.8, 8.0], device=self.device)
        kR = torch.tensor([4.0, 4.0, 1.5], device=self.device)
        kOmega = torch.tensor([1.0, 1.0, 0.5], device=self.device)

        mass = self.drone.masses.reshape(self.num_envs, -1)[..., :1]
        g = 9.81

        # -----------------------------------------------------------------------
        # STUDENT TODO (MPC 2/3): Implement your geometric controller.
        #
        # Variables available in this scope:
        #   drone_pos_flat        (N, 3)    drone position in env frame
        #   drone_rot             (N, 1, 4) drone orientation quaternion (w, x, y, z)
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in body frame
        # -----------------------------------------------------------------------
        # ----- ADD YOUR GEOMETRIC CONTROLLER CODE BELOW -----
        u1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        body_rate = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)

        return u1, body_rate

    def compute_mpc_action(self):
        if self.controller is None:
            raise RuntimeError(
                "MPC needs the rate controller"
            )
        if self.trajectory_mode != "planner":
            traj = self.plan_test_trajectory()
        else:
            traj = self.plan_local_trajectory()

        u1, body_rate = self.geometric_controller(
            traj["x_des"],
            traj["v_des"],
            traj["yaw_des"],
            a_des=traj["a_des"],
            yaw_rate_des=traj["yaw_rate_des"],
        )

        max_body_rates = self.controller.max_body_rates.to(self.device)
        max_thrust = self.controller.max_thrusts.sum(-1, keepdim=True).to(self.device)

        action_rate = (body_rate / max_body_rates).clamp(-1.0, 1.0)
        action_thrust = (2.0 * u1 / max_thrust - 1.0).clamp(-1.0, 1.0)
        return torch.cat([action_rate, action_thrust], dim=-1).unsqueeze(1)

    def _get_gate_center(self, gate_pos, gate_rot):
        """Compute gate center from gate origin (bottom-centre) by adding height/2 offset in gate frame.

        Args:
            gate_pos: (N, 3) gate origin positions in env frame.
            gate_rot: (N, 4) gate orientations (quaternions) in env frame.

        Returns:
            gate_center: (N, 3) gate centre positions in env frame.
        """
        offset_local = torch.tensor([0.0, 0.0, self.gate_height / 2.0], device=self.device)
        offset_local_expanded = offset_local.unsqueeze(0).expand(gate_pos.shape[0], -1)
        offset_world = quat_rotate(gate_rot, offset_local_expanded)
        return gate_pos + offset_world

    def _detect_gate_crossings(self, drone_pos_flat, current_gate_center, current_gate_rot,
                                gate_env_pos, gate_env_rot, batch_indices):
        """Detect whether the drone crossed through the current target gate this step.

        Uses plane-crossing (x: negative -> positive) plus bounding-box check
        (|y|, |z| < gate_width/2) in the gate frame centred at gate_center.

        Args:
            drone_pos_flat: (N, 3) current drone positions in env/world frame.
            current_gate_center: (N, 3) centre position of each env's current target gate
                in env/world frame.
            current_gate_rot: (N, 4) quaternion orientation of each env's current target gate
                in env/world frame.
            gate_env_pos: (N, G, 3) all gate origin positions for each env.
            gate_env_rot: (N, G, 4) all gate orientations (quaternions) for each env.
            batch_indices: (N,) environment indices used to gather the active gate per env.

        Side-effects: updates gate_indices, gate_passed, track_completed,
        prev_drone_in_gate_frame.

        Returns:
            gate_passed_this_step: (N,) bool — True for envs that just passed a gate.
            gate_index_changed:    (N,) bool — True for envs whose target gate advanced.
            new_gate_center:       (N, 3) — centre of the (possibly new) target gate.
        """
        drone_to_gate = drone_pos_flat - current_gate_center
        curr_in_gate = quat_rotate_inverse(current_gate_rot, drone_to_gate)  # (N, 3)

        # Drone position 
        prev_x = self.prev_drone_in_gate_frame[..., 0]
        curr_x = curr_in_gate[..., 0]
        crossed_plane = (prev_x < 0) & (curr_x > 0)

        # TODO: Build `gates_passed_successfully` as a per-environment binary mask.
        # Shape: (N,) where N == self.num_envs.
        # Type: bool is preferred (True/False); equivalent to 1/0 when cast to int/float.
        # Gate width and height can be accessed by using self.gate_width and self.gate_height.
        # ----- ADD YOUR GATE-CROSSING MASK CODE BELOW (replace the placeholder) -----
        gates_passed_successfully = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # ----- END STUDENT CODE -----
        
        gate_passed_this_step = gates_passed_successfully & (~self.gate_passed)
        self.gate_passed[gate_passed_this_step] = True

        old_gate_indices = self.gate_indices.clone()
        last_gate_passed = gate_passed_this_step & (self.gate_indices + 1 >= self.num_gates)
        self.track_completed[last_gate_passed] = True
        self.gate_indices[gate_passed_this_step] = torch.clamp(
            self.gate_indices[gate_passed_this_step] + 1,
            max=self.num_gates - 1,
        )
        gate_index_changed = (self.gate_indices != old_gate_indices)
        self.gate_passed[gate_index_changed] = False

        # Recompute gate centre for envs whose target changed
        new_gate_pos = gate_env_pos[batch_indices, self.gate_indices]
        new_gate_rot = gate_env_rot[batch_indices, self.gate_indices]
        new_gate_center = self._get_gate_center(new_gate_pos, new_gate_rot)
        new_in_gate = quat_rotate_inverse(new_gate_rot, drone_pos_flat - new_gate_center)

        # old_prev_in_gate_frame = self.prev_drone_in_gate_frame.clone()
        self.prev_drone_in_gate_frame = torch.where(
            gate_index_changed.unsqueeze(-1), new_in_gate, curr_in_gate,
        )

        return gate_passed_this_step, gate_index_changed, new_gate_center


    def _compute_reward_and_done(self):
        import traceback
        import sys

        track_completed = self.track_completed  # (N,)

        try:
            # Use cached pose tensors directly; self.drone_state does not include position/quaternion.
            drone_pos = self.drone.pos  # (N, 1, 3), env frame
            drone_rot = self.drone.rot  # (N, 1, 4), quaternion

            gate_world_pos, gate_world_rot = self.gates.get_world_poses()
            gate_env_pos, gate_env_rot = self.get_env_poses((gate_world_pos, gate_world_rot))

            # Print debug positions in the same frame (env frame) for easier sanity checks.
            gate_idx0 = int(self.gate_indices[0].item())
            # print(f"Drone position (env frame): {torch.round(drone_pos[0, 0] * 100) / 100}")
            if self.debug_gate_origins:
                self._draw_gate_origins(gate_world_pos, gate_world_rot, env_idx=0)

            batch_indices = torch.arange(self.num_envs, device=self.device)
            current_gate_pos = gate_env_pos[batch_indices, self.gate_indices]  # (N, 3)
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Failed in _compute_reward_and_done (num_envs={self.num_envs}, num_gates={self.num_gates})")
            print("=" * 80)
            traceback.print_exc()
            raise

        current_gate_rot = gate_env_rot[batch_indices, self.gate_indices]  # (N, 4)
        current_gate_center = self._get_gate_center(current_gate_pos, current_gate_rot)  # (N, 3)

        drone_pos_flat = drone_pos.squeeze(1)  # (N, 3)
        distance_to_gate = torch.norm(drone_pos_flat - current_gate_center, dim=-1)  # (N,)

        # --- gate crossing detection ---
        # You either _deteect_gate_crossings or _detect_gate_crossings_via_segments
        # This function call updates the gate indexes
        gate_passed_this_step, gate_index_changed, new_gate_center = self._detect_gate_crossings(
            drone_pos_flat, current_gate_center, current_gate_rot,
            gate_env_pos, gate_env_rot, batch_indices,
        )

        # -----------------------------------------------------------------------
        # STUDENT TODO (2/3): Implement your reward function.
        #
        # Variables available in this scope:
        #   drone_pos_flat        (N, 3)    drone position in env frame
        #   drone_rot             (N, 1, 4) drone orientation quaternion (w, x, y, z)
        #   distance_to_gate      (N,)      Euclidean distance to current gate centre
        #   current_gate_center   (N, 3)    3-D centre of the current target gate
        #   gate_passed_this_step (N,)      bool – True when drone just passed a gate
        #   gate_index_changed    (N,)      bool – True when target gate index advanced
        #   new_gate_center       (N, 3)    centre of the (possibly new) target gate
        #   crashed_collision     (N,)      bool – True when contact forces are detected
        #   self.drone.vel        (N, 1, 6) [lin_vel(3) | ang_vel(3)] in body frame
        #   self.gate_indices     (N,)      index of current target gate (0…num_gates-1)
        #   self.track_completed  (N,)      bool – True when the full lap is done
        #
        # Useful helpers (already imported at top of file):
        #   quat_axis(q, axis)          extract body axis vector (0=x, 1=y, 2=z)
        #   quat_rotate(q, v)           rotate vector v by quaternion q
        #   quat_rotate_inverse(q, v)   rotate vector v by inverse of quaternion q
        #
        # Your reward scalars are loaded in __init__ (STUDENT TODO 1/3) and
        # configured in cfg/task/DroneRace.yaml.
        # -----------------------------------------------------------------------
        # ----- ADD YOUR REWARD CODE BELOW (replace the placeholder) -----

        reward = torch.zeros(self.num_envs, device=self.device)  # (N,) ← replace this

        # ----- END STUDENT CODE -----

        # -----------------------------------------------------------------------
        # STUDENT TODO (3/3): Implement the crash / termination condition.
        # You might need to add geometric out-of-bounds checks.
        # -----------------------------------------------------------------------
        # ----- ADD YOUR CRASH CONDITION BELOW (replace the placeholder) -----

        crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # (N,) ← replace this

        # ----- END STUDENT CODE -----
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        completed_task = self.track_completed
        done = truncated | completed_task.unsqueeze(-1) | crashed.unsqueeze(-1)

        # --- stats ---
        self.stats["truncated"].add_(truncated.float())
        self.stats["collision"].add_(crashed.float().unsqueeze(-1))
        self.stats["success"].bitwise_or_(completed_task.unsqueeze(-1))
        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["gates_passed"][:] = (self.gate_indices + self.track_completed.long()).float().unsqueeze(1)
        # (optional) add extra stat tracking lines here if you add new metrics

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1).unsqueeze(-1),
                },
                "done": done,
                "terminated": crashed.unsqueeze(-1),
                "truncated": truncated,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
