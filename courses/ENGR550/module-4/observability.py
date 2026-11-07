"""
Module 4, Part 1: what the sensors can reveal (observability).

Two-state model with known inputs:
    x_{k+1} = F x_k + G u_k,   z_k = H x_k + n_k,   F = [[1, h], [0, 1]]

Noise-free outputs from x_0:  z_k = H F^k x_0 + (known input terms), so stacking
n of them gives  z - (known input terms) = O x_0  with
    O = [H; H F; ...; H F^{n-1}]
x_0 is uniquely determined exactly when O has full column rank n.

Position sensor, H = [1 0]:  O = [[1, 0], [1, h]]  (rank 2)
    z_0 = p_0,  z_1 = p_0 + h v_0 (+ h^2/2 u_0)  ->  v_0 = (z_1 - z_0 - h^2/2 u_0) / h.
    The model says position changes by exactly h v over a step, so the change between
    two position readings pins down velocity.
Velocity sensor, H = [0 1]:  O = [[0, 1], [0, 1]]  (rank 1)
    Velocity evolves as v_{k+1} = v_k + h u_k, which never involves p. Every reading is
    the same no matter where the vehicle started: the null space of O is [1, 0]^T, i.e.
    any shift of the initial position is invisible. Integrating velocity gives
    displacement, never absolute position.
"""

import numpy as np

H_STEP = 0.1       # s


def F(h):
    """Two-state transition matrix (Module 3)."""
    return np.array([[1.0, h],
                     [0.0, 1.0]])


def G(h):
    """Two-state input matrix for acceleration held over the step (Module 3)."""
    return np.array([h**2 / 2, h])


def observability_matrix(F_mat, H_mat):
    """O = [H; H F; ...; H F^{n-1}] for an n-state model. H may have one or several rows."""
    H_mat = np.atleast_2d(H_mat)
    n = F_mat.shape[0]
    blocks = []
    HFk = H_mat
    for _ in range(n):
        blocks.append(HFk)
        HFk = HFk @ F_mat
    return np.vstack(blocks)


def null_space(M, tol=1e-10):
    """Orthonormal basis for the null space of M (columns), from the SVD."""
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    return vt[rank:].T


def normalize_direction(v, tol=1e-10):
    """Scale a direction so its largest entry has magnitude 1 and its first nonzero entry is positive."""
    v = np.where(np.abs(v) < tol, 0.0, v / np.max(np.abs(v)))
    return v * np.sign(v[np.flatnonzero(v)[0]]) + 0.0    # + 0.0 turns -0.0 into 0.0


def rank_table(F_mat, cases, title):
    """Print rank, expected rank and unobservable directions for each sensor set."""
    n = F_mat.shape[0]
    print(title)
    print(f"  {'Sensors':<45} {'rank':>4} {'expected':>8}  unobservable direction")
    all_ok = True
    for name, H_mat, expected in cases:
        O = observability_matrix(F_mat, H_mat)
        r = np.linalg.matrix_rank(O)
        ns = null_space(O)
        direction = "none" if ns.shape[1] == 0 else ", ".join(
            np.array2string(normalize_direction(v), precision=2) for v in ns.T)
        ok = r == expected
        all_ok &= ok
        print(f"  {name:<45} {r:>4} {expected:>8}  {direction}{'' if ok else '   <-- MISMATCH'}")
    print(f"  (observable when rank = n = {n})\n")
    return all_ok


def recover_initial_state_demo(h=H_STEP):
    """Recover x_0 from two noise-free position readings with a known input."""
    x0 = np.array([2.0, -1.5])            # unknown to the 'estimator'
    u0 = 1.0                              # known input
    H = np.array([[1.0, 0.0]])
    z0 = (H @ x0)[0]
    z1 = (H @ (F(h) @ x0 + G(h) * u0))[0]
    O = observability_matrix(F(h), H)
    x0_hat = np.linalg.solve(O, np.array([z0, z1 - (H @ G(h))[0] * u0]))
    print("Demo: two position readings determine the initial state")
    print(f"  true x0 = {x0},  z0 = {z0:.4f},  z1 = {z1:.4f},  known u0 = {u0}")
    print(f"  solve O x0 = [z0, z1 - H G u0]  ->  x0 = {np.round(x0_hat, 10)}")

    shift = np.array([5.0, 0.0])
    Hv = np.array([[0.0, 1.0]])
    zs = [(Hv @ np.linalg.matrix_power(F(h), k) @ x0)[0] for k in range(5)]
    zs_shift = [(Hv @ np.linalg.matrix_power(F(h), k) @ (x0 + shift))[0] for k in range(5)]
    print("Demo: velocity readings cannot see a 5 m shift in initial position")
    print(f"  readings from x0          : {np.round(zs, 4)}")
    print(f"  readings from x0 + [5, 0] : {np.round(zs_shift, 4)}\n")


def main():
    h = H_STEP
    Fh = F(h)
    print(f"h = {h} s,  F =\n{Fh}\n")
    print("O for position sensing (expected [[1, 0], [1, h]]):")
    print(observability_matrix(Fh, [1.0, 0.0]), "\n")

    ok = rank_table(Fh, [
        ("Position          H = [1 0]", np.array([1.0, 0.0]), 2),
        ("Velocity          H = [0 1]", np.array([0.0, 1.0]), 1),
        ("Position+velocity H = I2", np.eye(2), 2),
    ], "Rank checks, two-state model")

    s = np.linalg.svd(observability_matrix(Fh, [1.0, 0.0]), compute_uv=False)
    print(f"Side note: singular values of O for position sensing are {np.round(s, 4)}.")
    print("  The small one is about h/sqrt(2): velocity is observable, but only through a small")
    print("  position change per step, so noise is amplified by ~1/h (the Module 1 experiment).\n")

    recover_initial_state_demo(h)
    print("All rank checks match." if ok else "RANK MISMATCH.")


if __name__ == "__main__":
    main()
