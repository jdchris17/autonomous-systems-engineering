# Planar Tracking with Inertial Navigation: Engineering Report

Module 16 capstone. Regenerate all tables, figures and logs with one command from this folder:

```
python run_all.py
```

The run takes about 7 s. Outputs go to `outputs/`: `config_used.json`, `scenario_table.md/.csv`,
`multiseed.csv`, `results.txt`, three figures, and `logs/` with per-tick state, covariance and
control logs and per-update innovation/NIS logs for every scenario.

---

## 1. Architecture and assumptions

```
reference (timed)  ─►  guidance + LQI speed control  ─►  u_a, u_ω (clipped)  ─►  vehicle (truth)
        ▲                       ▲                                                   │
        │                       └──── EKF estimate [p, v, ψ, b_a, b_g] ◄── IMU 20 Hz, position 1 Hz, wheel 10 Hz
        └─ noise-free, Module 15 model, saved per timestamp: p_ref, v_ref, ψ_ref, a_ref, ω_ref
```

| Part | Model and assumptions |
|---|---|
| **Vehicle (truth)** | Module 15 discrete planar model (h = 0.05 s, level, no slip).<br>Longitudinal drag: a = u_a − c_true·v + d_a + w_a.<br>Yaw: ω = g·u_ω + d_ω + w_ω.<br>Motion disturbances: SD 0.2 m/s² and 0.02 rad/s.<br>Nominal values: c_true = 0.2, g = 1. |
| **Sensors** | IMU at 20 Hz measures the actual acceleration and yaw rate, plus bias (0.05 m/s², 0.005 rad/s) and noise (0.1, 0.01).<br>Position at 1 Hz, 0.5 m per axis.<br>Wheel speed at 10 Hz, 0.1 m/s. |
| **Estimator** | Module 15 six-state EKF, unchanged. It uses the IMU as the input and does not model drag.<br>Q contains IMU noise only.<br>Initial offsets [1 m, −1 m, 0.3 m/s, 10°]; biases start at 0 with SDs 0.1 and 0.02. |
| **Guidance** | Position error resolved into the reference frame (along e∥, across e⊥).<br>v_target = clip(v_ref + k∥·e∥, 0.5, 4).<br>ψ_target = ψ_ref + atan2(k⊥·e⊥, v_ref). |
| **Speed control** | Module 12 LQI gains (R_c = 1) with feedforward a_ref + c_nom·v_target.<br>Command limited to ±2 m/s², with conditional integration. |
| **Heading control** | u_ω = ω_ref + k_ψ·wrap(ψ_target − ψ̂), limited to ±0.5 rad/s. |
| **Timing per tick** | 1. Correct with readings at t_k.<br>2. Compute commands.<br>3. Propagate truth.<br>4. Generate that interval's IMU readings.<br>5. Predict the EKF to t_{k+1}. |

**Main assumptions:**
- The controller sees only the reference and the EKF output. Truth is used only for simulation and scoring, and by the true-state benchmark.
- Every scenario shares the same underlying noise draws. Each generates its readings from its own trajectory.
- Cross-track error is scored from **true** position, resolved in the reference frame.

## 2. Frozen design

| Item | Value | Note |
|---|---|---|
| k∥, k⊥, k_ψ | 0.3 s⁻¹, 0.3 s⁻¹, 1 s⁻¹ | starting values, unchanged |
| LQI k_e, k_ξ | 1.2208, −0.4939 | Module 12, R_c = 1; designed at h = 0.02 s, reused at 0.05 s |
| Limits | \|u_a\| ≤ 2 m/s², \|u_ω\| ≤ 0.5 rad/s, v_target ∈ [0.5, 4] m/s | |
| Seeds | Demo seed 2; seeds 0–9 for the nominal and outage runs | |
| **Targets, frozen before the final suite** | Nominal tracking RMSE < 1 m; outage peak tracking error < 3 m | Peak scored from 18 s **to the end of the run**, so the recovery and the 2 m fixes are included. Every seed must pass. |

**Tuning record:** one nominal check on seed 2 gave a tracking RMSE of 0.39 m. The true-state benchmark, with the same gains, gave 0.03 m. The remaining error is therefore navigation error, which no guidance gain can remove, so I made no adjustment. The configuration was frozen in `config.py` before the scenario suite and the multi-seed runs.

## 3. Measured results

**All scenarios, seed 2** (full table in `outputs/scenario_table.md`):

| Scenario | Tracking RMSE | Peak error ≥ 18 s | Cross-track RMSE / max | Time at yaw / accel limit | Nav position RMSE | Max position SD 18–28 s |
|---|---|---|---|---|---|---|
| True-state benchmark | 0.03 m | 0.07 m | 0.01 / 0.04 m | 0 / 0 s | 0.36 m | 0.38 m |
| Nominal | 0.39 m | 0.36 m | 0.30 / 0.77 m | 0.40 / 0 s | 0.36 m | 0.39 m |
| Disturbance | 0.42 m | 0.55 m | 0.31 / 0.77 m | 0.40 / 0 s | 0.36 m | 0.39 m |
| Sensor failure | 0.40 m | 0.36 m | 0.32 / 0.77 m | 0.40 / 0 s | 0.37 m | **1.44 m** |
| Model mismatch | 0.47 m | 0.40 m | 0.36 / 0.91 m | 0.60 / 0 s | 0.36 m | 0.39 m |
| **Authority limit** | **84.9 m** | **132.8 m** | **58.2 / 131.1 m** | **39.4** / 0.55 s | 0.43 m | 0.42 m |

**Multi-seed results, seeds 0–9** (`outputs/multiseed.csv`, `multiseed.png`):

| Scenario | Metric | Mean | Worst (seed) | Target | Passed |
|---|---|---|---|---|---|
| Nominal | Tracking RMSE | 0.49 m | 0.70 m (9) | < 1 m | 10/10 |
| Sensor failure | Peak tracking error from 18 s | 1.03 m | 2.49 m (6) | < 3 m | 10/10 |

**Numerical health:** no failures in any run. All values stayed finite. The smallest covariance eigenvalue was ≥ 1.7e-7. Covariance asymmetry was ≤ 1e-15.

**Navigation (seed 2):**
- Heading-estimation RMSE is about 3°, almost all of it from the initial 10° error in the first 3 s.
- Speed-estimation RMSE is 0.032 m/s.
- Final b_a error is +0.004 ± 0.004 m/s². Final b_g error is −0.0013 ± 0.0008 rad/s.

**Interpretation:**
1. **Tracking is limited by navigation.** In every working scenario, tracking RMSE (0.39–0.47 m) is about equal to navigation error (0.36 m), while the benchmark reaches 0.03 m. The controller follows the estimate closely. The estimate is what is off.
2. **The disturbance is absorbed.** d_a = −0.3 is handled by the LQI integrator, and d_ω = 0.03 by the heading loop. The peak tracking error rises only from 0.36 to 0.55 m.
3. **Model mismatch costs effort, not accuracy.**
   - The drag of 0.35 gets no feedforward, so acceleration effort triples (8.4 → 24.3) as the integrator supplies the extra force.
   - The yaw gain of 0.8 is corrected by the heading loop.
   - The EKF is unaffected because it never used a drag model.
4. **The outage widens the filter's uncertainty but rarely its error.** Over the 10 s outage the filter's 2σ grows to 2.9 m. Seed 2's actual error stays about 0.3 m, so the filter is conservative there. Across seeds the outage peak varies sevenfold (0.36–2.49 m). Seed 2 is the *easiest* draw, so the multi-seed table, not the demo seed, is the evidence for the 3 m target.
5. **A brief yaw saturation at the start (0.4 s) in every estimated-state run** comes from the 10° initial heading error in the estimate. It is not caused by the reference.

## 4. Failure case: insufficient turning authority

The yaw command is limited to ±0.05 rad/s, but the reference turns at 0.2 rad/s (a 10 m radius at 2 m/s). At 2 m/s, 0.05 rad/s gives a turning radius of 40 m. The timed path is therefore **infeasible** at any speed the controller allows, because v_target ≥ 0.5 m/s would still need 0.1 rad/s.

**Evidence:**
- **Yaw command:** at its limit for 39.4 of 40 s (panel d of `diagnostics.png`).
- **Cross-track error:** grows without bound, to a maximum of 131 m. Tracking RMSE is 84.9 m.
- **Trajectory:** the vehicle traces a wide arc away from the S (`trajectories.png`).
- **Speed control makes it worse:** the along-track law raises v_target to catch the timed reference. Higher speed makes the turning radius *larger*.
- **Acceleration:** saturates at −2 m/s² near 31 s when the geometry reverses the along-track error.
- **Navigation is unaffected:** position RMSE is 0.43 m. The estimator knows exactly where the vehicle is; the controller cannot get it to the reference. This is a pure control-authority failure, which the separate navigation and tracking scores make obvious.

This is not a gain problem. No feedback gain creates turning rate the actuator does not have.

## 5. Next change, justified by these results

**1. Bound heading drift during position outages.** This is the largest risk to a target that currently passes.
- Seed 6 entered the outage with a −3.5° heading error and a gyro-bias error of +0.008 rad/s.
- Module 15's straight-line budget predicts v·T·δψ + ½·v·δb_g·T² ≈ 1.2 + 0.8 = 2.0 m of cross-track drift over 10 s.
- Measured navigation error was 2.5 m, giving a 2.49 m tracking peak against the 3 m target.
- The peak occurs 0.3 s after fixes resume, so it is drift accumulated during the outage, not something the controller introduced.
- A heading aid that keeps working during the outage would bound this term directly: Module 14's landmark bearing, or a magnetometer.
- **Cheaper alternative:** keep turning before outages. Turning makes heading and gyro bias observable; this course's trajectory runs straight for the 3 s before 18 s.

**2. Add a feasibility check on the reference** before tracking it: |ω_ref| ≤ u_ω,max with margin. When the check fails:
- re-time the reference (slow down until v·κ ≤ u_ω,max), or
- re-plan it to a feasible curvature.

Optionally, schedule v_target down when the yaw command saturates rather than speeding up.

**Not justified by the results:** retuning the guidance gains. With the benchmark at 0.03 m, they are not the limiting factor.
