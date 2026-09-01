"""module-3 -- inertia_tensor.py  (Computational Exercise: Build an Inertia Tensor)

Rectangular rigid body, dimensions a, b, c, mass m. About the centre of mass
the principal moments are

    I_xx = m/12 (b^2 + c^2)
    I_yy = m/12 (a^2 + c^2)
    I_zz = m/12 (a^2 + b^2)

Three cases:

  1. The diagonal tensor built directly from those formulas. Eigenvalues are
     the diagonal entries; eigenvectors are the coordinate axes.

  2. A hand-written non-diagonal symmetric tensor (nonzero products of
     inertia). Eigen-decompose it.

  3. The same body physically rotated by a known R. Its tensor is
     I' = R I R^T, still symmetric, now full. eigh must recover the original
     principal moments and the columns of R (the principal axes).

For every case we check  V^T I V = D  is diagonal, where the columns of V are
the eigenvectors. Physically: V is the orientation of the body's principal
axes, and in that frame the rotational dynamics decouple -- a torque-free spin
about a principal axis stays a pure spin, with no wobble.
"""

import numpy as np
import matplotlib.pyplot as plt

from rotations import rot_x, rot_y, rot_z


def box_principal_moments(m, a, b, c):
    return np.array([m / 12 * (b**2 + c**2),
                     m / 12 * (a**2 + c**2),
                     m / 12 * (a**2 + b**2)])


def eigendecompose(I, label):
    """Symmetric eigen-decomposition + the V^T I V = D check. Returns (w, V)."""
    w, V = np.linalg.eigh(I)                 # eigh: symmetric, orthonormal V
    D = V.T @ I @ V
    off = np.abs(D - np.diag(np.diag(D))).max()
    print(f"--- {label} ---")
    print(f"I =\n{np.array2string(I, precision=4, suppress_small=True)}")
    print(f"eigenvalues (principal moments) : {np.array2string(w, precision=5)}")
    print("eigenvectors (principal axes, columns):")
    print(np.array2string(V, precision=4, suppress_small=True))
    print(f"V^T V - I  (orthonormal?)       : {np.abs(V.T @ V - np.eye(3)).max():.2e}")
    print(f"max off-diagonal of V^T I V     : {off:.2e}   -> D is diagonal")
    print(f"D diag                          : {np.array2string(np.diag(D), precision=5)}")
    print()
    return w, V


def main():
    np.set_printoptions(suppress=True)
    m, a, b, c = 12.0, 3.0, 2.0, 1.0          # mass, x-, y-, z-extent
    Ip = box_principal_moments(m, a, b, c)
    print("=" * 74)
    print(f"BOX: m = {m}, dimensions a,b,c = {a},{b},{c}")
    print(f"principal moments  m/12*(...)  = {np.array2string(Ip, precision=4)}")
    print("=" * 74)

    # 1. diagonal --------------------------------------------------------
    I_diag = np.diag(Ip)
    eigendecompose(I_diag, "case 1: diagonal tensor")
    print("  -> eigenvalues ARE the diagonal; eigenvectors ARE x, y, z. The box")
    print("     is already aligned with its principal axes.\n")

    # 2. hand-written symmetric, non-diagonal --------------------------
    I_hand = np.array([[8.0, -2.0,  1.0],
                       [-2.0, 10.0, -0.5],
                       [1.0, -0.5,  6.0]])
    w2, V2 = eigendecompose(I_hand, "case 2: hand-written symmetric tensor")
    print("  -> the products of inertia (off-diagonal terms) are the price of")
    print("     describing the body in the 'wrong' axes. eigh finds axes where")
    print("     they vanish.\n")

    # 3. physically rotated box ---------------------------------------
    R_true = rot_z(np.deg2rad(35)) @ rot_x(np.deg2rad(20))
    I_rot = R_true @ I_diag @ R_true.T
    w3, V3 = eigendecompose(I_rot, "case 3: box rotated by a known R (35 deg z, 20 deg x)")

    # match eigenvector columns to R_true columns (order + sign are free)
    match = np.abs(V3.T @ R_true)
    perm = np.argmax(match, axis=0)
    aligned = np.array([np.sign((V3[:, perm[k]] @ R_true[:, k])) * V3[:, perm[k]]
                        for k in range(3)]).T
    print(f"  recovered principal moments : {np.array2string(np.sort(w3), precision=5)}")
    print(f"  original principal moments  : {np.array2string(np.sort(Ip), precision=5)}")
    print(f"  |recovered axes - R_true|   : {np.abs(aligned - R_true).max():.2e}")
    print("  -> eigh recovered both the principal moments AND the body's")
    print("     orientation. That is what a principal-axis frame IS.")
    print("=" * 74)

    _plot(I_diag, R_true)


def _cuboid_edges(half):
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    verts = s * half
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(s[i] - s[j])) == 2:      # differ in one coord
                edges.append((verts[i], verts[j]))
    return edges


def _plot(I_diag, R):
    half = np.array([3.0, 2.0, 1.0]) / 2
    fig = plt.figure(figsize=(11, 5))

    for k, (Rk, title) in enumerate([(np.eye(3), "aligned: I is diagonal"),
                                     (R, "rotated: I is full, principal axes tilted")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for p, q in _cuboid_edges(half):
            p2, q2 = Rk @ p, Rk @ q
            ax.plot(*zip(p2, q2), color="0.5", lw=1)
        colors = ["tab:red", "tab:green", "tab:blue"]
        for axis in range(3):
            v = Rk[:, axis] * (half[axis] + 1.2)
            ax.quiver(0, 0, 0, *v, color=colors[axis], lw=2)
            ax.text(*(v * 1.1), f"e{axis+1}", color=colors[axis])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x_I"); ax.set_ylabel("y_I"); ax.set_zlabel("z_I")

    fig.suptitle("Principal axes are the eigenvectors of the inertia tensor")
    fig.tight_layout()
    fig.savefig("inertia_tensor.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
