"""
Experiment C: support vector machines (scikit-learn SVC).

All inputs are the engineered features standardized with TRAINING statistics.
Part 1  Two-feature view (log width, roundness): linear SVM margin, support vectors, and the
        logistic-regression boundary on the same axes.
Part 2  Soft margin: C in {0.001 ... 100} for the linear SVM, 2 and 5 features.
Part 3  RBF kernel: grid over C and gamma; 2-D boundaries; 5-feature validation selection.
Engineering checks: is the RBF gain genuine nonlinearity (compare logistic regression with
quadratic features, and train-vs-validation gaps) and how does the support-vector count scale.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from classifiers import OUT, LogisticRegressionGD, Standardizer
from star_data import engineered, standard_splits

TWO = [1, 2]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
RBF_C = [0.1, 1, 10, 100]
RBF_G = [0.03, 0.3, 3, 30]


def prepare():
    sp = standard_splits()
    E = {k: engineered(v[0]) for k, v in sp.items()}
    st = Standardizer().fit(E["train"])
    return {k: st.transform(v) for k, v in E.items()}, {k: v[1] for k, v in sp.items()}


def quad_features(Z):
    cols = [Z]
    p = Z.shape[1]
    for i in range(p):
        for j in range(i, p):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)


def acc(m, X, y):
    return float(np.mean(m.predict(X) == y))


def main():
    OUT.mkdir(exist_ok=True)
    Z, y = prepare()
    Z2 = {k: v[:, TWO] for k, v in Z.items()}
    print("Experiment C: support vector machines (standardized engineered features)")

    # ----------------------------------------------------------------- Part 1
    lin = SVC(kernel="linear", C=1.0).fit(Z2["train"], y["train"])
    w, b = lin.coef_.ravel(), lin.intercept_[0]
    lr = LogisticRegressionGD(lam=1e-3).fit(Z2["train"], y["train"])
    ang = np.degrees(np.arccos(abs(w @ lr.theta[1:]) / np.linalg.norm(w) / np.linalg.norm(lr.theta[1:])))
    print("\n  Part 1 - 2 features (log width, roundness), linear SVM C = 1")
    print(f"    w = {np.round(w, 3)}, b = {b:.3f}, margin width 2/||w|| = {2 / np.linalg.norm(w):.3f}"
          f" (standardized units)")
    print(f"    support vectors: {lin.n_support_.sum()} of {y['train'].size}"
          f" ({lin.n_support_.sum() / y['train'].size:.0%}); bound (alpha = C, inside margin or"
          f" misclassified): {np.sum(np.isclose(np.abs(lin.dual_coef_), 1.0))}")
    print(f"    test accuracy: SVM {acc(lin, Z2['test'], y['test']):.3f},"
          f" logistic {np.mean(lr.predict(Z2['test']) == y['test']):.3f};"
          f" angle between the two boundaries {ang:.1f} deg")
    plot_part1(Z2, y, lin, lr)

    # ----------------------------------------------------------------- Part 2
    print("\n  Part 2 - soft margin, linear SVM")
    print(f"    {'C':>7} | {'2D train':>8}{'2D test':>8}{'#SV':>6}{'margin':>8} | {'5D train':>8}"
          f"{'5D test':>8}{'#SV':>6}{'margin':>8}{'fit s':>7}")
    soft = {}
    for C in C_GRID:
        row = []
        for X in (Z2, Z):
            t = time.perf_counter()
            m = SVC(kernel="linear", C=C, max_iter=2_000_000).fit(X["train"], y["train"])
            dt = time.perf_counter() - t
            row.append((acc(m, X["train"], y["train"]), acc(m, X["test"], y["test"]),
                        m.n_support_.sum(), 2 / np.linalg.norm(m.coef_), dt, m))
        soft[C] = row
        a, b5 = row
        print(f"    {C:>7g} | {a[0]:>8.3f}{a[1]:>8.3f}{a[2]:>6}{a[3]:>8.2f} | {b5[0]:>8.3f}{b5[1]:>8.3f}"
              f"{b5[2]:>6}{b5[3]:>8.2f}{b5[4]:>7.1f}")
    plot_part2(Z2, y, soft)

    # ----------------------------------------------------------------- Part 3
    print("\n  Part 3 - RBF kernel: 5 features, validation accuracy [train accuracy] (#SV)")
    print("    gamma ->" + "".join(f"{g:>22g}" for g in RBF_G))
    grid = {}
    for C in RBF_C:
        cells = []
        for g in RBF_G:
            m = SVC(kernel="rbf", C=C, gamma=g).fit(Z["train"], y["train"])
            grid[(C, g)] = (acc(m, Z["val"], y["val"]), acc(m, Z["train"], y["train"]),
                            m.n_support_.sum())
            v, tr, nsv = grid[(C, g)]
            cells.append(f"{v:.3f} [{tr:.3f}] ({nsv})")
        print(f"    C={C:<6g}" + "".join(f"{c:>22}" for c in cells))
    best = max(grid, key=lambda k: grid[k][0])
    bm = SVC(kernel="rbf", C=best[0], gamma=best[1]).fit(Z["train"], y["train"])
    print(f"    selected on validation: C = {best[0]}, gamma = {best[1]}  ->  test accuracy"
          f" {acc(bm, Z['test'], y['test']):.3f}, #SV {bm.n_support_.sum()}")
    plot_part3(Z2, y)

    # ----------------------------------------------------------------- engineering questions
    print("\n  Is the nonlinearity genuine? (5 features, test accuracy)")
    lin5 = soft[1][1][5]
    lrq = LogisticRegressionGD(lam=1e-3).fit(quad_features(Z["train"]), y["train"])
    print(f"    linear SVM (C=1)                           {acc(lin5, Z['test'], y['test']):.3f}")
    print(f"    logistic, linear features                  "
          f"{np.mean(LogisticRegressionGD(lam=1e-3).fit(Z['train'], y['train']).predict(Z['test']) == y['test']):.3f}")
    print(f"    logistic, + all squares and products (20)  {np.mean(lrq.predict(quad_features(Z['test'])) == y['test']):.3f}")
    print(f"    RBF SVM (validated)                        {acc(bm, Z['test'], y['test']):.3f}"
          f"   train {acc(bm, Z['train'], y['train']):.3f}")
    worst = (100, 30)
    print(f"    RBF SVM C=100, gamma=30 (overfit check)    val {grid[worst][0]:.3f}"
          f"   train {grid[worst][1]:.3f}")

    print("\n  Support vectors vs training-set size (5 features)")
    print(f"    {'n_train':>8}{'linear C=1':>14}{'RBF selected':>16}{'RBF C=100,g=30':>18}")
    for n in (500, 1000, 2000, 4000):
        Xn, yn = Z["train"][:n], y["train"][:n]
        counts = [SVC(kernel=k, C=c, gamma=g).fit(Xn, yn).n_support_.sum()
                  for k, c, g in (("linear", 1, "scale"), ("rbf", best[0], best[1]), ("rbf", 100, 30))]
        print(f"    {n:>8}" + "".join(f"{c:>8} ({c / n:>3.0%})".rjust(w)
                                       for c, w in zip(counts, (14, 16, 18))))
    for f in ("exp_c_linear_margin.png", "exp_c_soft_margin.png", "exp_c_rbf_grid.png"):
        print(f"  saved {OUT / f}")


def grid2d():
    xx, yy = np.meshgrid(np.linspace(-3, 3.5, 300), np.linspace(-3.2, 1.6, 300))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def scatter_bg(ax, Z2, y, n=1500):
    X, t = Z2["test"][:n], y["test"][:n]
    ax.scatter(X[t == 0, 0], X[t == 0, 1], s=4, alpha=0.35, color="tab:red")
    ax.scatter(X[t == 1, 0], X[t == 1, 1], s=4, alpha=0.35, color="tab:blue")


def plot_part1(Z2, y, lin, lr):
    xx, yy, G = grid2d()
    fig, ax = plt.subplots(figsize=(8, 6))
    X, t = Z2["train"], y["train"]
    ax.scatter(X[t == 0, 0], X[t == 0, 1], s=5, alpha=0.3, color="tab:red", label="artifact")
    ax.scatter(X[t == 1, 0], X[t == 1, 1], s=5, alpha=0.3, color="tab:blue", label="star")
    sv = lin.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=18, facecolor="none", edgecolor="k", lw=0.3, alpha=0.4,
               label=f"support vectors ({len(sv)})")
    D = lin.decision_function(G).reshape(xx.shape)
    ax.contour(xx, yy, D, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    P = lr.predict_proba(G).reshape(xx.shape)
    ax.contour(xx, yy, P, levels=[0.5], colors="tab:green", linewidths=2, linestyles="-.")
    ax.plot([], [], "k-", lw=2, label="SVM boundary (f = 0)")
    ax.plot([], [], "k--", lw=1, label="SVM margins (f = +-1)")
    ax.plot([], [], "-.", color="tab:green", lw=2, label="logistic p = 0.5")
    ax.set_xlabel("log width (standardized)")
    ax.set_ylabel("roundness (standardized)")
    ax.set_title("Linear SVM (C = 1): margin and support vectors vs logistic regression")
    ax.legend(fontsize=8, loc="lower left", markerscale=2)
    ax.set_xlim(-3, 3.5)
    ax.set_ylim(-3.2, 1.6)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_linear_margin.png", dpi=140)
    plt.close(fig)


def plot_part2(Z2, y, soft):
    xx, yy, G = grid2d()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    scatter_bg(ax, Z2, y)
    cmap = plt.get_cmap("viridis")
    for k, C in enumerate(C_GRID):
        m = soft[C][0][5]
        D = m.decision_function(G).reshape(xx.shape)
        col = cmap(k / (len(C_GRID) - 1))
        ax.contour(xx, yy, D, levels=[0], colors=[col], linewidths=2)
        ax.contour(xx, yy, D, levels=[-1, 1], colors=[col], linewidths=0.7, linestyles="--")
        ax.plot([], [], color=col, label=f"C = {C:g} (margin {soft[C][0][3]:.2f}, {soft[C][0][2]} SV)")
    ax.set_xlim(-3, 3.5)
    ax.set_ylim(-3.2, 1.6)
    ax.set_xlabel("log width (standardized)")
    ax.set_ylabel("roundness (standardized)")
    ax.set_title("Soft margin: boundary (solid) and margins (dashed) vs C")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    for j, (lab, ls) in enumerate((("2 features", "--"), ("5 features", "-"))):
        ax.semilogx(C_GRID, [soft[C][j][0] for C in C_GRID], "s" + ls, color="tab:blue",
                    label=f"{lab}: train")
        ax.semilogx(C_GRID, [soft[C][j][1] for C in C_GRID], "o" + ls, color="tab:red",
                    label=f"{lab}: test")
    ax2 = ax.twinx()
    ax2.semilogx(C_GRID, [soft[C][1][2] for C in C_GRID], "^:", color="gray", label="#SV (5 features)")
    ax2.set_ylabel("support vectors", color="gray")
    ax.set_xlabel("C")
    ax.set_ylabel("accuracy")
    ax.set_title("Training and test accuracy vs C (linear SVM)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_soft_margin.png", dpi=140)
    plt.close(fig)


def plot_part3(Z2, y):
    xx, yy, G = grid2d()
    fig, axes = plt.subplots(len(RBF_C), len(RBF_G), figsize=(17, 14), sharex=True, sharey=True)
    for i, C in enumerate(RBF_C):
        for j, g in enumerate(RBF_G):
            ax = axes[i, j]
            m = SVC(kernel="rbf", C=C, gamma=g).fit(Z2["train"], y["train"])
            D = m.decision_function(G).reshape(xx.shape)
            ax.contourf(xx, yy, D > 0, levels=[-0.5, 0.5, 1.5], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
            scatter_bg(ax, Z2, y, 800)
            ax.contour(xx, yy, D, levels=[0], colors="k", linewidths=1.2)
            ax.set_title(f"C={C:g}, gamma={g:g}: val {acc(m, Z2['val'], y['val']):.3f},"
                         f" train {acc(m, Z2['train'], y['train']):.3f}, SV {m.n_support_.sum()}",
                         fontsize=8)
            if i == len(RBF_C) - 1:
                ax.set_xlabel("log width (std)")
            if j == 0:
                ax.set_ylabel("roundness (std)")
    fig.suptitle("RBF SVM on 2 features: blue = predicted star. Small gamma ~ smooth, large gamma ~"
                 " islands around individual points")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_rbf_grid.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
