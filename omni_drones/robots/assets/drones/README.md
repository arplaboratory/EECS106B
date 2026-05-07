# Drone asset bundles

| Directory | Entry USD | URDF | Notes |
|-----------|-----------|------|--------|
| `five_in/` | `5_in_drone.usd` | `urdf/5_in_drone.urdf` | 5-inch reference; scale `(1,1,1)`. |

Shared: `configuration/*.usd`, `meshes/*.dae`. Import the IsaacLab cfg from `omni_drones.robots.assets.five_in_drone` (`FIVE_IN_DRONE`).

Parameter YAML for the omni_drones `MultirotorBase` adapter lives at `five_in/five_in_drone.yaml` (loaded by `omni_drones.robots.drone.five_in.FiveIn`).
