# Iris vs 5-inch racing quad

A side-by-side derivation showing **why the racing drone is ~10× more agile**
than the standard Iris multirotor, even though both are X-configuration
quadrotors driven by the same `Allocation` and `RateController` primitives in
this repo.

The numbers in this doc are pulled directly from
`omni_drones/robots/assets/usd/iris.yaml` and
`omni_drones/robots/assets/drones/five_in/five_in_drone.yaml` (see also
`racing-drone.md` for FiveIn's URDF/inertia table). The figure rendered by
`scripts/plot_drone_comparison.py` is at
`docs/figures/drone_comparison_iris_vs_five_in.png`.

![Iris vs FiveIn](figures/drone_comparison_iris_vs_five_in.png)

## 1. Per-rotor model

Each rotor produces a body-frame thrust force and a yaw-axis reaction torque
that are quadratic in the rotor's angular velocity $\omega_i$:

$$
T_i \;=\; k_f \, \omega_i^2,
\qquad
\tau_{yaw,i} \;=\; \pm\, k_m \, \omega_i^2
$$

with $k_f$ the **thrust coefficient** and $k_m$ the **moment coefficient**.
The sign on $\tau_{yaw,i}$ alternates by rotor direction (CW / CCW) so that
the four together can synthesize a net yaw torque from differential RPM.

| symbol      | Iris            | FiveIn          | ratio |
|-------------|-----------------|-----------------|-------|
| $k_f$       | $8.55\times10^{-6}$ | $4.80\times10^{-7}$ | 0.06× |
| $k_m$       | $1.37\times10^{-7}$ | $2.00\times10^{-9}$ | 0.01× |
| $\omega_{\max}$ | 838 rad/s       | 5000 rad/s     | 5.97× |

FiveIn's individual rotor is "weaker per rev" (small $k_f$, small $k_m$, smaller
prop) — but it spins **6× faster**, and thrust scales as $\omega^2$, so the
square wins.

## 2. Total thrust and TWR

The four rotors produce a total body-frame thrust along $+z_b$:

$$
T_{tot}(\omega) \;=\; \sum_{i=1}^{4} k_f \, \omega_i^2
\;\;\xRightarrow{\omega_i = \omega_{\max}}\;\;
T_{\max} \;=\; 4\, k_f\, \omega_{\max}^2
$$

The **thrust-to-weight ratio** governs how much margin the controller has
above gravity:

$$
\mathrm{TWR} \;=\; \frac{T_{\max}}{m\,g}
$$

Plugging in:

| quantity    | Iris       | FiveIn     | ratio |
|-------------|------------|------------|-------|
| $T_{\max}$  | 24.0 N     | 48.0 N     | 2.0×  |
| $T_{hover} = mg$ | 14.9 N | 4.91 N     | 0.33× |
| **TWR**     | **1.61×**  | **9.79×**  | **6.1×** |

Iris is barely powered enough to climb (TWR ≈ 1.6 means only ~60% of gravity
in headroom). FiveIn has **almost an order of magnitude of headroom**, which
is why racing pilots flick into vertical climbs and recover from gravity-tower
manoeuvres. The peak vertical acceleration is

$$
a_z^{\max} \;=\; \frac{T_{\max}}{m} - g
\;=\; (\mathrm{TWR} - 1)\, g
$$

so Iris peaks at **0.61 g** of vertical acceleration, FiveIn at **8.79 g**.

## 3. Control allocation (X configuration)

Both drones use the X-config allocation matrix that maps individual-rotor
thrusts $\mathbf{f} = [f_1, f_2, f_3, f_4]^\top$ (with $f_i = k_f\,\omega_i^2$)
to the body-frame wrench $[F_z, \tau_x, \tau_y, \tau_z]^\top$:

$$
\begin{bmatrix} F_z \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}
\;=\;
\underbrace{
\begin{bmatrix}
1 & 1 & 1 & 1 \\
\frac{L}{\sqrt{2}} & -\frac{L}{\sqrt{2}} & -\frac{L}{\sqrt{2}} & \frac{L}{\sqrt{2}} \\
-\frac{L}{\sqrt{2}} & -\frac{L}{\sqrt{2}} & \frac{L}{\sqrt{2}} & \frac{L}{\sqrt{2}} \\
\frac{k_m}{k_f} & -\frac{k_m}{k_f} & \frac{k_m}{k_f} & -\frac{k_m}{k_f}
\end{bmatrix}}_{\displaystyle \mathbf{A}}
\begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \end{bmatrix}
$$

The implementation lives in `omni_drones/robots/dynamics/allocation.py`. $L$
is the frame "tip-to-tip" arm length (so $L/\sqrt{2}$ is the planar moment
arm in X configuration).

| symbol      | Iris   | FiveIn | ratio |
|-------------|--------|--------|-------|
| $L$ (avg)   | 0.247 m | 0.177 m | 0.71× |
| $L/\sqrt{2}$ (eff. arm) | 0.175 m | 0.125 m | 0.71× |

The arm difference is small — the airframes look similar from above. The
agility gap comes from elsewhere.

## 4. Peak body torques and angular acceleration

Take roll: maximum lateral asymmetry is achieved when two rotors on one side
spin at $\omega_{\max}$ and the other two are off. That gives

$$
\tau_{x}^{\max} \;=\; 2\, k_f\, \omega_{\max}^2 \,\frac{L}{\sqrt{2}}
$$

Yaw similarly uses the full differential of the moment-coef pair:

$$
\tau_{z}^{\max} \;=\; 4\, k_m\, \omega_{\max}^2
$$

and Newton's rotational equation $\boldsymbol\tau = \mathbf{I}\,\dot{\boldsymbol\omega}$
gives the peak achievable angular acceleration along each principal axis:

$$
\alpha_x^{\max} \;=\; \frac{\tau_x^{\max}}{I_{xx}},
\qquad
\alpha_z^{\max} \;=\; \frac{\tau_z^{\max}}{I_{zz}}
$$

This is the punchline:

| quantity            | Iris        | FiveIn      | ratio  |
|---------------------|-------------|-------------|--------|
| $I_{xx}$            | 0.0348 kg·m² | 0.003 kg·m²  | 0.09×  |
| $\tau_x^{\max}$     | 2.10 N·m    | 3.00 N·m    | 1.43×  |
| **$\alpha_x^{\max}$** | **60 rad/s²**  | **999 rad/s²**  | **16.6×** |
| $I_{zz}$            | 0.0977 kg·m² | 0.006 kg·m²  | 0.06×  |
| $\alpha_z^{\max}$   | 3.93 rad/s² | 33.3 rad/s² | 8.5×   |

**FiveIn rolls 16.6× faster than Iris** despite having only 1.43× more roll
torque, because its moment of inertia is **11× smaller**. Inertia scales like
$m \cdot L^2$ — racing drones are deliberately tiny *and* light.

## 5. Why the same controller works for both

The PPO observation/action layout in `DroneRaceEnv` is the same regardless of
which drone is plugged in (`drone_model.name: Iris` vs `Hummingbird`). What
changes between airframes is:

1. **The mapping** $\omega_i \mapsto T_i, \tau_{yaw,i}$ (the rotor model coeffs)
2. **The allocation matrix** $\mathbf{A}$ (via $L$ and $k_m / k_f$)
3. **The mass and inertia** in the equations of motion

All three live in YAML and get loaded by `MultirotorBase.make()` — no policy
or MDP changes are required to swap drones, only a retraining run because the
*time-scale* of the dynamics changes by an order of magnitude.

## 6. Summary

|                       | Iris (general purpose)        | FiveIn (5-inch racer)         |
|-----------------------|-------------------------------|-------------------------------|
| design intent         | docile, large, low-bandwidth  | aggressive, light, high-bandwidth |
| TWR                   | 1.6×                          | 9.8×                          |
| peak vertical accel   | 0.6 g                         | 8.8 g                         |
| peak roll accel       | 60 rad/s²                     | 999 rad/s²                    |
| peak yaw accel        | 3.9 rad/s²                    | 33.3 rad/s²                   |
| natural frequency*    | ~2 Hz                         | ~10 Hz                        |
| typical use case      | hobby flight / nav benchmarks | racing / freestyle / SkyDreamer |

\* Approx. attitude-loop bandwidth implied by $\sqrt{\alpha_{\max}/\theta_{\max}}$
   for a unit-amplitude command — the racer can close the loop ~5× faster.

This is why the **same dynamics primitives** in `omni_drones/robots/dynamics/`
(Allocation, Motor, AttitudeController, BodyRateController) cover both
airframes by parameter swap, but a policy trained on Iris does *not* transfer
zero-shot to FiveIn — the time constants are too different.
