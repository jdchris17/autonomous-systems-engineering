"""Navigation uncertainty: a 2D position estimate with a full covariance
matrix, its eigen-structure, and the covariance ellipse."""

import matplotlib.pyplot as plt
import numpy as np

MU = np.array([100.0, 200.0])
SIGMA = np.array([[16.0, 8.0], [8.0, 25.0]])
N_SAMPLES = 5000


def covariance_ellipse(mu, eigenvalues, eigenvectors, k, n_points=200):
    """Boundary of the k-sigma ellipse: mu + k * V * sqrt(L) * unit circle."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    return mu[:, None] + k * eigenvectors @ (np.sqrt(eigenvalues)[:, None] * circle)


if __name__ == "__main__":
    # 1-3. Marginal spreads and correlation come straight from SIGMA.
    std_e = np.sqrt(SIGMA[0, 0])
    std_n = np.sqrt(SIGMA[1, 1])
    rho = SIGMA[0, 1] / (std_e * std_n)

    # 4-6. Eigen-decomposition (eigh: SIGMA is symmetric). eigh returns
    # ascending eigenvalues, so flip to put the principal axis first.
    eigenvalues, eigenvectors = np.linalg.eigh(SIGMA)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    principal = eigenvectors[:, 0]
    if principal[0] < 0:
        principal = -principal
        eigenvectors[:, 0] = principal
    angle_deg = np.degrees(np.arctan2(principal[1], principal[0]))

    print("Position estimate x = [E, N], mean = [100, 200] m")
    print(f"Covariance matrix (m^2):\n{SIGMA}\n")
    print(f"1. East standard deviation:   {std_e:.4f} m")
    print(f"2. North standard deviation:  {std_n:.4f} m")
    print(f"3. East/north correlation:    {rho:.4f}")
    print(f"4. Eigenvalues (m^2):         {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
    print(f"   -> std along each axis:    {np.sqrt(eigenvalues[0]):.4f} m, {np.sqrt(eigenvalues[1]):.4f} m")
    print("5. Eigenvectors (columns):")
    print(f"   v1 = [{eigenvectors[0, 0]: .4f}, {eigenvectors[1, 0]: .4f}]  (lambda1 = {eigenvalues[0]:.4f})")
    print(f"   v2 = [{eigenvectors[0, 1]: .4f}, {eigenvectors[1, 1]: .4f}]  (lambda2 = {eigenvalues[1]:.4f})")
    print(f"6. Principal uncertainty direction: {angle_deg:.2f} deg from East toward North")

    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(MU, SIGMA, size=N_SAMPLES)

    sample_cov = np.cov(samples.T)
    print(f"\nSample covariance from {N_SAMPLES:,} draws:\n{np.round(sample_cov, 2)}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.3, color="steelblue", label="samples")
    for k, style in zip([1, 2, 3], ["-", "--", ":"]):
        ellipse = covariance_ellipse(MU, eigenvalues, eigenvectors, k)
        ax.plot(ellipse[0], ellipse[1], style, color="firebrick", linewidth=1.5, label=f"{k}-sigma ellipse")
    for i, color in zip(range(2), ["black", "dimgray"]):
        v = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.annotate(
            "",
            xy=MU + v,
            xytext=MU,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
        )
    ax.plot(*MU, "k+", markersize=12)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Navigation uncertainty: covariance ellipse (arrows = eigenvectors x 1-sigma)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("navigation_uncertainty.png", dpi=150)
    print("\nSaved plot to navigation_uncertainty.png")

    print(
        f"""
What this means
---------------
"Position error is about 4-5 m" is not a complete description. It quotes the two
marginal spreads ({std_e:.0f} m east, {std_n:.0f} m north) and throws away the rest:
  - The errors are correlated (rho = {rho:.2f}): when the estimate is too far east
    it tends to also be too far north, so the cloud is a tilted ellipse, not
    a circle or an axis-aligned box.
  - The eigen-decomposition gives the ellipse's true axes. The worst direction
    is {angle_deg:.0f} deg from East with std {np.sqrt(eigenvalues[0]):.2f} m; the best
    (perpendicular) direction has only {np.sqrt(eigenvalues[1]):.2f} m. Neither equals 4 or 5.
  - Uncertainty depends on direction, so a single number cannot say how
    wrong you might be along the road you care about. The covariance matrix
    can: for a unit direction u, the std along it is sqrt(u' Sigma u).
That is why navigation systems carry the full covariance matrix."""
    )
