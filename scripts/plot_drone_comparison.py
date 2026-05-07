#!/usr/bin/env python3
# Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
"""Plot a side-by-side comparison of the Iris and 5-inch racing drone.

Reads the per-airframe parameters baked into this repo:
  - omni_drones/robots/assets/usd/iris.yaml         (Iris)
  - omni_drones/robots/assets/drones/five_in/
        five_in_drone.yaml                          (FiveIn)
  - racing-drone.md                                  (FiveIn inertia)

Computes hover/peak thrust, thrust-to-weight ratio, peak upward and angular
accelerations, and saves
``docs/figures/drone_comparison_iris_vs_five_in.png``. Run from repo root.
"""

from __future__ import annotations

import math
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRAVITY = 9.81

# --- Iris (multirotor base, omni_drones repo defaults) ----------------------
IRIS = {
    "name": "Iris",
    "color": "#4C72B0",       # cool blue
    "mass": 1.52,             # kg
    "I_xx": 0.0348, "I_yy": 0.0459, "I_zz": 0.0977,   # kg·m²
    # arm_lengths in iris.yaml are 4-vec; X-config planar moment arm = avg/√2
    "arm_lengths": [0.255539, 0.238537, 0.255539, 0.238537],
    "k_f": 8.54858e-6,        # thrust coef, T_i = k_f · ω_i²  (N·s²)
    "k_m": 1.3677728816219314e-7,  # moment coef, τ_yaw_i = k_m · ω_i² (N·m·s²)
    "omega_max": 838.0,       # rad/s
}

# --- FiveIn (5-inch racing quad, racing-drone.md table + URDF) --------------
FIVE_IN = {
    "name": "FiveIn (racing)",
    "color": "#C44E52",       # red
    "mass": 0.5,
    "I_xx": 0.003, "I_yy": 0.003, "I_zz": 0.006,
    # arm_length already encoded as L (= 2·arm_xy in URDF) in five_in_drone.yaml
    "arm_lengths": [0.1766] * 4,
    "k_f": 4.8e-7,
    "k_m": 2.0e-9,
    "omega_max": 5000.0,
}


def derive(d: dict) -> dict:
    """Augment a drone-spec dict with derived quantities."""
    L_avg = float(np.mean(d["arm_lengths"]))
    arm = L_avg / math.sqrt(2.0)            # X-config moment arm

    T_max = 4 * d["k_f"] * d["omega_max"] ** 2
    T_hover = d["mass"] * GRAVITY
    twr = T_max / T_hover
    omega_hover = math.sqrt(T_hover / (4 * d["k_f"]))
    a_up = T_max / d["mass"] - GRAVITY      # m/s² peak upward
    a_up_g = a_up / GRAVITY                 # in g

    # Peak body torques: full diff between front and back rotor pair, X-config.
    tau_max_roll = 2 * d["k_f"] * d["omega_max"] ** 2 * arm
    tau_max_pitch = tau_max_roll                           # symmetric
    tau_max_yaw = 4 * d["k_m"] * d["omega_max"] ** 2

    alpha_roll = tau_max_roll / d["I_xx"]
    alpha_pitch = tau_max_pitch / d["I_yy"]
    alpha_yaw = tau_max_yaw / d["I_zz"]

    return {
        **d,
        "L_avg": L_avg, "arm_eff": arm,
        "T_max": T_max, "T_hover": T_hover, "TWR": twr,
        "omega_hover": omega_hover,
        "a_up": a_up, "a_up_g": a_up_g,
        "tau_max_roll": tau_max_roll,
        "tau_max_yaw": tau_max_yaw,
        "alpha_roll": alpha_roll,
        "alpha_pitch": alpha_pitch,
        "alpha_yaw": alpha_yaw,
    }


def _bars(ax, label, vals, drones, log=False, fmt="{:.2f}"):
    """One grouped horizontal bar per drone."""
    y = np.arange(len(drones))
    ax.barh(y, vals, color=[d["color"] for d in drones], edgecolor="k",
            alpha=0.9, height=0.65)
    if log:
        ax.set_xscale("log")
    for yi, v, d in zip(y, vals, drones):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=9,
                color="k")
    ax.set_yticks(y)
    ax.set_yticklabels([d["name"] for d in drones], fontsize=9)
    ax.set_title(label, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    iris = derive(IRIS)
    five = derive(FIVE_IN)
    drones = [iris, five]

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drone_comparison_iris_vs_five_in.png"

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    fig.suptitle(
        "Iris vs 5-inch racing quad — physical parameters & performance limits",
        fontsize=13, fontweight="bold",
    )

    # --- Row 1: airframe physical parameters ---
    _bars(axes[0, 0], "Mass [kg]", [d["mass"] for d in drones], drones)
    _bars(axes[0, 1], "Diagonal inertia I_xx [kg·m²]",
          [d["I_xx"] for d in drones], drones, log=True, fmt="{:.4g}")
    _bars(axes[0, 2], "Effective moment arm L/√2 [m]",
          [d["arm_eff"] for d in drones], drones, fmt="{:.3f}")

    # --- Row 2: rotor model ---
    _bars(axes[1, 0], "Thrust coef k_f [N·s²]",
          [d["k_f"] for d in drones], drones, log=True, fmt="{:.2e}")
    _bars(axes[1, 1], "ω_max [rad/s]",
          [d["omega_max"] for d in drones], drones, fmt="{:.0f}")
    _bars(axes[1, 2], "Peak total thrust T_max [N]",
          [d["T_max"] for d in drones], drones, fmt="{:.1f}")

    # --- Row 3: derived performance ---
    _bars(axes[2, 0], "Thrust-to-Weight Ratio (TWR)",
          [d["TWR"] for d in drones], drones, fmt="{:.2f}×")
    _bars(axes[2, 1], "Peak vertical accel [g]",
          [d["a_up_g"] for d in drones], drones, fmt="{:.2f}g")
    _bars(axes[2, 2], "Peak roll/pitch ang. accel [rad/s²]",
          [d["alpha_roll"] for d in drones], drones, log=True, fmt="{:.0f}")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # ---- ASCII summary table ------------------------------------------------
    print()
    print(f"{'quantity':<32} {'Iris':>14} {'FiveIn':>14}  ratio")
    print("-" * 80)
    rows = [
        ("mass [kg]",            iris["mass"],         five["mass"]),
        ("I_xx [kg·m²]",         iris["I_xx"],         five["I_xx"]),
        ("L/√2 [m]",             iris["arm_eff"],      five["arm_eff"]),
        ("k_f [N·s²]",           iris["k_f"],          five["k_f"]),
        ("k_m [N·m·s²]",         iris["k_m"],          five["k_m"]),
        ("ω_max [rad/s]",        iris["omega_max"],    five["omega_max"]),
        ("ω_hover [rad/s]",      iris["omega_hover"],  five["omega_hover"]),
        ("T_max [N]",            iris["T_max"],        five["T_max"]),
        ("T_hover [N]",          iris["T_hover"],      five["T_hover"]),
        ("TWR",                  iris["TWR"],          five["TWR"]),
        ("peak vertical accel [g]", iris["a_up_g"],    five["a_up_g"]),
        ("peak τ_roll [N·m]",    iris["tau_max_roll"], five["tau_max_roll"]),
        ("α_roll [rad/s²]",      iris["alpha_roll"],   five["alpha_roll"]),
        ("α_yaw [rad/s²]",       iris["alpha_yaw"],    five["alpha_yaw"]),
    ]
    for name, vi, vf in rows:
        ratio = vf / vi if vi != 0 else float("inf")
        print(f"{name:<32} {vi:>14.4g} {vf:>14.4g}  ×{ratio:>5.2f}")


if __name__ == "__main__":
    main()
