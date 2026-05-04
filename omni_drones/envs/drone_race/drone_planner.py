from __future__ import annotations

from dataclasses import dataclass, field
import math

import casadi as ca
import numpy as np


@dataclass
class GateObstacle:
    cx: float = 0.0
    cy: float = 0.0
    gate_yaw: float = 0.0
    gate_scale: float = 1.0
    gate_height: float = 2.5
    gate_width: float = 2.5


@dataclass
class PlannerParams:
    """Parameters for the minimum-time local gate planner."""

    # Total number of direct-transcription intervals.
    N: int = 40
    # Output sampling period used by the controller.
    dt: float = 0.01
    # Action block horizon
    horizon: int = 2

    min_segment_time: float = 0.05
    max_segment_time: float = 6.0

    v_max: float = 6.0
    a_xy_max: float = 8.0
    a_z_max: float = 6.0
    a_z_min: float = -9.81

    gate_height: float = 2.5
    gate_width: float = 2.5
    mass: float = 1.52
    Jx: float = 0.0347563
    Jy: float = 0.0458929
    Jz: float = 0.0977

    # STUDENT TODO: add more as you see fit


@dataclass
class PlannerResult:
    success: bool
    x_des: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    v_des: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    a_des: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    yaw_des: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    yaw_rate_des: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    dt: float = 0.01
    first_gate_time: float = 0.0
    total_time: float = 0.0
    solver_stats: dict = field(default_factory=dict)


class DronePlanner:

    def __init__(self, params: PlannerParams | None = None):
        self.params = params or PlannerParams()

    def solve(
        self,
        start: tuple[float, float, float, float, float, float, float],
        obstacles: list[CircularObstacle] | None = None,
    ) -> PlannerResult:
        p = self.params
        N = p.N
        obstacles = obstacles or []

        opti = ca.Opti()

        ## Decision variables
        X = opti.variable(4, N + 1)  # flat outputs [x; y; z; yaw] at each node
        U = opti.variable(4, N)      # [u1; body_rate] at each interval
        T = opti.variable()          # total trajectory time
        dt = p.dt                    # derived timestep

        ## TODO: Objective — minimize total trajectory time

        ## TODO: Dynamics constraints — Euler integration of unicycle model

        ## TODO: Boundary constraints — pin start and goal states

        ## TODO: Control bounds — bound v and omega

        ## TODO: Time bounds — bound total time T (Force to be positive)

        ## TODO: Obstacle avoidance — keep all nodes outside each obstacle

        ## TODO: Initial guesses

        opti.solver(
            "ipopt",
            {
                "expand": True,
                "print_time": False,   # CasADi timing output
            },
            {
                "max_iter": 500,
                "print_level": 0,      # IPOPT internal verbosity
                "sb": "yes",           # <- This is to silence IPOPT
                "tol": 1e-4,
                "acceptable_tol": 1e-3,
            },
        )

        try:
            sol = opti.solve()
        except RuntimeError as exc:
            print(f"Solver failed: {e}")
            debug = opti.debug

        ## TODO: Build PlannerResult from sol and return
        result = PlannerResult(
            success=True,
            p_des=p_des[:p.horizon],
            v_des=v_des[:p.horizon],
            a_des=a_des[:p.horizon],
            yaw_des=yaw_des[:p.horizon],
            yaw_rate_des=yaw_rate_des[:p.horizon],
            dt=p.dt,
            first_gate_time=float(0.0),
            total_time=float(0.0),
        )
        return result