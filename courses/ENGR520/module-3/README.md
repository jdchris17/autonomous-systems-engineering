# Module 3 — Rigid-body attitude: frames, rotations, Euler angles

Rotational kinematics of a rigid body: the inertia tensor and its principal
axes, rotation matrices in SO(3), the body/inertial frame distinction, the
aerospace Euler-angle convention, and gimbal lock.

## Conventions (fixed here, used everywhere — never left implicit)

**Frames**

| Frame | Symbol | Meaning |
|---|---|---|
| Inertial | `I` | non-rotating reference frame; overall motion is described here |
| Body | `B` | frame glued to the rigid body; rotates with it |

**Rotation matrix**

`R_IB` is the body's attitude — the rotation that takes **body coordinates to
inertial coordinates**:

```
v_I = R_IB @ v_B
v_B = R_IB.T @ v_I          (because R^-1 = R^T for R in SO(3))
```

The **columns of `R_IB`** are the body axes `x_B, y_B, z_B` expressed in
inertial coordinates.

**Elementary rotations** (`rotations.py`)

`rot_x, rot_y, rot_z` are **active, right-handed** rotations of a *vector*
about a fixed coordinate axis, angle in **radians**. (Their transpose rotates
the frame instead.)

**Euler angles** — aerospace **Z–Y–X (3–2–1)**, intrinsic:

```
R_IB(phi, theta, psi) = rot_z(psi) @ rot_y(theta) @ rot_x(phi)

phi   = roll   about body x     (applied last)
theta = pitch  about body y
psi   = yaw    about body z     (applied first)
```

Singular at `theta = ±90°` (gimbal lock).

**Quaternions** (`quaternions.py`)

```
layout   : scalar-first,  q = [w, x, y, z]
product  : Hamilton (not JPL)
meaning  : a unit q encodes the same rotation as R_IB  (body -> inertial)
           v_I = R(q) v_B = q (x) [0, v_B] (x) q*
double cover : q and -q are the same rotation
```

Attitude kinematics use the **body-frame** angular velocity:

```
q_dot = 0.5 * q (x) [0, omega_B]
```

(If `omega` were in the inertial frame it would be `0.5 * [0, omega_I] (x) q`;
if `q` mapped `I -> B` the order/sign would flip. We use the form above only.)

## Files

| File | Section | What it does |
|---|---|---|
| `rotations.py` | 7, 8 | Library: `rot_x/rot_y/rot_z`, `axis_angle_to_R` (Rodrigues), plus `is_rotation`, `orthonormality_error`, `determinant_error`, `assert_rotation`. Imported by everything else. |
| `check_rotations.py` | 8 | Verifies `R^T R = I` and `det R = +1` over many angles; rotates known vectors and checks by hand; shows length/angle preservation, `R^-1 = R^T`, and non-commutativity. |
| `frames.py` | 6 | Body vs inertial. Same physical vector, two coordinate sets; a body-fixed marker on a spinning tilted body — constant in `B`, circling in `I`. Figure `frames.png`. |
| `euler_angles.py` | 10 | `euler_zyx_to_R` and `R_to_euler_zyx` (with a gimbal-lock guard); 20 000-sample round-trip test; the columns-are-body-axes reading. |
| `gimbal_lock.py` | 11 | The `theta -> 90°` experiment. (A) the Euler-rate matrix `A(phi,theta)` with `det A = cos theta`, so `cond(A) -> inf`; (B) `dR/dphi` and `dR/dpsi` becoming collinear. Recovers angles right at the lock — only `yaw - roll` survives. Figure `gimbal_lock.png`. |
| `inertia_tensor.py` | 5 | Box principal moments `m/12·(...)`; eigen-decomposition of (1) the diagonal tensor, (2) a hand-written symmetric tensor, (3) a box rotated by a known `R`. Checks `V^T I V = D` is diagonal; case 3 recovers both the principal moments and the orientation. Figure `inertia_tensor.png`. |
| `quaternions.py` | 13 | Library, implemented from scratch: `quaternion_multiply` (Hamilton), `quaternion_conjugate`, `quaternion_normalize`, `axis_angle_to_quaternion`, `quaternion_to_rotation_matrix`, plus `rotate_vector`, `quaternion_angle`. |
| `check_quaternions.py` | 13 | `R(q)^T R(q) = I`, `det R(q) = +1`; `axis-angle -> q -> R(q)` equals `axis-angle -> R` directly (Rodrigues) to ~1e-16; `R(q1 ⊗ q2) = R(q1) R(q2)`; `q ~ -q`; `q*` is the inverse rotation; `rotate_vector == R @ v`. |
| `quaternion_kinematics.py` | 14 | Defines `quaternion_derivative(q, omega_body) = 0.5 q ⊗ [0, omega]`; checks norm preservation, `omega` round-trip, agreement with `R_dot = R skew(omega_B)`, and shows the silent error from feeding `omega` in the wrong frame. |
| `integrate_omega.py` | 15 | `omega = [0,0,1]`, `q(0) = 1`, RK4 + renormalize each step. Converts `q(t) -> R(t)`, rotates body x into inertial → traces `(cos t, sin t, 0)` to ~1e-11. Also shows forward-Euler norm drift with/without the `q <- q/‖q‖` step. Figure `integrate_omega.png`. |

### Major Project 1 — 6-DOF rigid-body simulator

| File | Section | What it does |
|---|---|---|
| `rigid_body.py` | 18 | The simulator. 13-state `x = [r, v, q, omega]`; `RigidBody(mass, inertia)` with `derivative(t, x)` implementing `r_dot = v`, `v_dot = F/m`, `q_dot = 0.5 q ⊗ [0,omega]`, `omega_dot = I^-1(tau - omega × I omega)`. RK4 integrator, quaternion renormalized each step. Diagnostics: rotational KE, angular momentum in body and inertial frames. |
| `sim_torque_axis.py` | 20 | **Validation.** Start at rest, constant `tau = [0,0,tau_z]` about a principal axis → `omega_z(t) = tau_z/I_z · t` exactly, no coupling. Numerical vs analytic agree to ~1e-12. Figure `sim_torque_axis.png`. |
| `sim_torque_free.py` | 21 | **Invariants.** `tau = 0`, arbitrary `omega0`. `T = ½ omega^T I omega` and `|L|` conserved to ~1e-14; `L` is a fixed vector in the inertial frame while its body components oscillate. Figure `sim_torque_free.png`. |
| `sim_intermediate_axis.py` | 22 | **Tennis-racket / Dzhanibekov.** Body with `I1 < I2 < I3`, spun about each axis with a tiny seed. Axes 1 and 3 are stable; axis 2 is unstable — measured growth rate `1.994/s` vs predicted `1.999/s` — and the body flips over repeatedly. No tumbling logic is coded; it falls out of Euler's equations. Figure `sim_intermediate_axis.png`. |
| `visualize_body.py` | 23 | `omega(t)`, then `R_IB(t)` applied to the body axes and plotted as inertial components, plus a strip of 3-D box snapshots and (if a writer is available) `visualize_body.gif`. Figures `visualize_body_timeseries.png`, `visualize_body_snapshots.png`. |

## Running

```bash
cd physics/module-3
python check_rotations.py
python frames.py
python euler_angles.py
python gimbal_lock.py
python inertia_tensor.py
python check_quaternions.py
python quaternion_kinematics.py
python integrate_omega.py
# Major Project 1
python sim_torque_axis.py
python sim_torque_free.py
python sim_intermediate_axis.py
python visualize_body.py
```

Each script prints a labelled summary of its results; most also save a PNG
next to the script.

## Key ideas

- **A rotation matrix is orthogonal because rotation is rigid.** Its columns
  are the rotated basis vectors, still orthonormal, so `R^T R = I`. That is
  also why re-expressing a vector between frames never changes its length —
  the coordinates move, the physical vector does not.

- **`det R = +1`, not −1.** `+1` is a proper rotation; `−1` would include a
  reflection and flip handedness. The elementary matrices and any product of
  them stay in `SO(3)`.

- **Order matters.** `rot_x @ rot_y ≠ rot_y @ rot_x`, which is exactly why an
  Euler-angle convention has to be stated in full: axis sequence, intrinsic
  vs extrinsic, active vs passive, and which frame maps to which.

- **Principal axes are eigenvectors of the inertia tensor.** In that frame the
  products of inertia vanish and `I` is diagonal, so a torque-free spin about a
  principal axis stays a pure spin. `eigh` finds that frame from any symmetric
  `I`.

- **Gimbal lock is a coordinate problem, not a physical one.** At `theta = 90°`
  the map `(phi, theta, psi) -> R` loses rank: roll and yaw become the same
  motion and only their combination is recoverable. The rigid body still has
  all three rotational degrees of freedom — the `(phi, theta, psi)` chart is
  what went singular.

- **Quaternions dodge that.** A unit quaternion covers all of `SO(3)` with no
  singular orientation; `q_dot = 0.5 q ⊗ [0, omega_B]` has no `1/cos theta`.
  The price is a redundant fourth number that must be kept normalized (`q <-
  q/‖q‖`) and the `q ~ -q` double cover. `integrate_omega.py` shows a constant
  `omega` integrating to the exact analytical spin with error ~1e-11.

- **State the quaternion convention.** Scalar-first vs scalar-last, Hamilton vs
  JPL, `B->I` vs `I->B`, body vs inertial `omega` — each flips a sign or an
  order somewhere. Mixing two of them is one of the most common attitude-code
  bugs; `quaternion_kinematics.py` demonstrates the silent error.

- **The physics is in the equations, not in special cases.** The rigid-body
  simulator has one `derivative()` function. Linear spin-up, torque-free
  precession, and the Dzhanibekov flip are all the *same* four equations with
  different initial conditions — exactly as the falling object, the orbit, and
  the escape trajectory were one ODE in the earlier modules. Conserved
  quantities (`T`, `|L|`) are the check that the integrator is trustworthy
  before any of the interesting behaviour is believed.
