"""
Experiment B: perceptron and logistic regression from scratch.

Inputs are the engineered features (log peak, log width, roundness, log background, log SNR),
standardized with TRAINING statistics. Fixed splits: 4000 train / 2000 val / 4000 test.

Part 1  Perceptron: (a) two-feature, stars vs hot pixels only (almost separable),
        (b) two-feature, stars vs all artifacts, (c) all five features.
Part 2  Logistic regression with BCE in softplus form, trained by Module 2's gradient_descent,
        small L2 penalty lam = 1e-3 on the weights (not the bias).
Part 3  Gradient check with Module 2's numerical_gradient, then scikit-learn LogisticRegression
        with matched objective: sklearn minimises 0.5||w||^2 + C sum(loss), i.e. after dividing by
        C n: mean(loss) + ||w||^2 / (2 C n)  ->  C = 1 / (n lam); intercept unpenalized; lbfgs, tol 1e-10.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from classifiers import (OUT, LogisticRegressionGD, Perceptron, Standardizer, add_bias, confusion,
                         logistic_gradient, logistic_loss, numerical_gradient, rates,
                         relative_discrepancy)
from star_data import ENG_NAMES, engineered, standard_splits

LAM = 1e-3
TWO = [1, 2]          # log width, roundness


def prepare():
    sp = standard_splits()
    E = {k: engineered(v[0]) for k, v in sp.items()}
    st = Standardizer().fit(E["train"])
    Z = {k: st.transform(v) for k, v in E.items()}
    y = {k: v[1] for k, v in sp.items()}
    sub = {k: v[2] for k, v in sp.items()}
    return Z, y, sub


def boundary_line(ax, w, b, xlim, **kw):
    xs = np.linspace(*xlim, 50)
    ax.plot(xs, -(b + w[0] * xs) / w[1], **kw)


def main():
    OUT.mkdir(exist_ok=True)
    Z, y, sub = prepare()
    print("Experiment B: perceptron and logistic regression (features standardized on train)")

    # ----------------------------------------------------------------- Part 1
    print("\n  Part 1 - perceptron (lr = 1, shuffled each epoch, up to 50 epochs)")
    m_hp = {k: (sub[k] == "hot pixel") | (y[k] == 1) for k in Z}
    cases = {
        "2 features, stars vs hot pixels": (Z["train"][m_hp["train"]][:, TWO], y["train"][m_hp["train"]],
                                            Z["test"][m_hp["test"]][:, TWO], y["test"][m_hp["test"]]),
        "2 features, all candidates": (Z["train"][:, TWO], y["train"], Z["test"][:, TWO], y["test"]),
        "5 features, all candidates": (Z["train"], y["train"], Z["test"], y["test"]),
    }
    percs = {}
    for name, (Xtr, ytr, Xte, yte) in cases.items():
        p = Perceptron(epochs=50, seed=1).fit(Xtr, ytr)
        percs[name] = p
        print(f"    {name:<34} epochs {len(p.mistakes):>2}, mistakes first/last epoch"
              f" {p.mistakes[0]:>4}/{p.mistakes[-1]:>4} (of {ytr.size}),"
              f" train acc {np.mean(p.predict(Xtr) == ytr):.3f}, test acc {np.mean(p.predict(Xte) == yte):.3f}")
        print(f"      final weights [b, w...] = {np.round(p.final_weights, 2)}")

    # ----------------------------------------------------------------- Part 2
    print(f"\n  Part 2 - logistic regression from scratch (lam = {LAM}, step 1/L, ||grad|| <= 1e-7)")
    lrs = {}
    for name, cols in (("2 features", TWO), ("5 features", list(range(5)))):
        m = LogisticRegressionGD(lam=LAM).fit(Z["train"][:, cols], y["train"])
        lrs[name] = (m, cols)
        acc_te = np.mean(m.predict(Z["test"][:, cols]) == y["test"])
        print(f"    {name}: {m.info['status']} after {m.info['iterations']} updates (step {m.step:.3f}),"
              f" final loss {m.loss_history[-1]:.5f}, train acc"
              f" {np.mean(m.predict(Z['train'][:, cols]) == y['train']):.3f}, test acc {acc_te:.3f}")
        print(f"      theta [b, w...] = {np.round(m.theta, 4)}")
    print("      5-feature weights by feature: " + ", ".join(
        f"{n} {w:+.2f}" for n, w in zip(ENG_NAMES, lrs["5 features"][0].theta[1:])))

    # ----------------------------------------------------------------- Part 3
    print("\n  Part 3 - gradient check (relative discrepancy E, 5-feature loss incl. L2 term)")
    Xb = add_bias(Z["train"])
    yt = y["train"].astype(float)
    rng = np.random.default_rng(5)
    thetas = {"zeros": np.zeros(6), "trained": lrs["5 features"][0].theta,
              "random": rng.normal(0, 1, 6), "large random": rng.normal(0, 8, 6)}
    hs = [1e-2, 1e-4, 1e-6, 1e-8]
    print(f"    {'theta':<14}" + "".join(f"{f'h={h:g}':>11}" for h in hs))
    for tn, th in thetas.items():
        ga = logistic_gradient(Xb, yt, th, LAM)
        J = lambda t: logistic_loss(Xb, yt, t, LAM)  # noqa: E731
        print(f"    {tn:<14}" + "".join(
            f"{relative_discrepancy(ga, numerical_gradient(J, th, h)):>11.1e}" for h in hs))

    mine = lrs["5 features"][0]
    n = Z["train"].shape[0]
    sk = LogisticRegression(C=1.0 / (n * LAM), tol=1e-10, max_iter=10_000).fit(Z["train"], y["train"])
    th_sk = np.concatenate([sk.intercept_, sk.coef_.ravel()])
    p_m, p_s = mine.predict_proba(Z["test"]), sk.predict_proba(Z["test"])[:, 1]
    print(f"\n    scikit-learn LogisticRegression(C = 1/(n lam) = {1 / (n * LAM):.3f}, lbfgs, tol 1e-10)")
    print(f"      theta (sklearn) = {np.round(th_sk, 4)}")
    print(f"      |theta_mine - theta_sklearn| = {np.linalg.norm(mine.theta - th_sk):.2e},"
          f" max |p_mine - p_sklearn| on test = {np.max(np.abs(p_m - p_s)):.2e},"
          f" identical test predictions: {np.mean((p_m >= 0.5) == (p_s >= 0.5)):.4f}")
    print(f"      training objective: mine {logistic_loss(Xb, yt, mine.theta, LAM):.10f},"
          f" sklearn {logistic_loss(Xb, yt, th_sk, LAM):.10f}")

    # ----------------------------------------------------------------- confusion matrices
    print("\n  Test-set confusion matrices [[TN, FP], [FN, TP]] (5 features, threshold 0.5 / sign):")
    p5 = percs["5 features, all candidates"]
    for name, yhat in (("perceptron", p5.predict(Z["test"])), ("logistic", mine.predict(Z["test"]))):
        r = rates(y["test"], yhat)
        print(f"    {name:<11}{confusion(y['test'], yhat).tolist()}  accuracy {r['accuracy']:.3f},"
              f" precision {r['precision']:.3f}, recall {r['recall']:.3f}")

    plots(Z, y, sub, percs, lrs, m_hp)


def plots(Z, y, sub, percs, lrs, m_hp):
    # learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    for name, p in percs.items():
        n = (m_hp["train"].sum() if "hot" in name else y["train"].size)
        ax.plot(np.arange(1, len(p.mistakes) + 1), np.array(p.mistakes) / n, "o-", ms=3,
                label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mistakes / training examples")
    ax.set_yscale("log")
    ax.set_title("Perceptron learning curves (mistakes per epoch)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for name, (m, _) in lrs.items():
        lh = m.loss_history
        ax.plot(np.arange(lh.size), lh - lh[-1] + 1e-12, label=f"{name} (final {lh[-1]:.4f})")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.set_yscale("log")
    ax.set_xlabel("gradient-descent iteration")
    ax.set_ylabel("loss - final loss")
    ax.set_title("Logistic regression learning curves (BCE + L2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_learning_curves.png", dpi=140)
    plt.close(fig)

    # 2-D decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (title, mask_tr) in zip(axes, (("stars vs hot pixels", m_hp["test"]),
                                           ("stars vs all artifacts", np.ones_like(y["test"], bool)))):
        Xt = Z["test"][mask_tr][:, TWO]
        yt = y["test"][mask_tr]
        ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1], s=4, alpha=0.35, color="tab:red", label="artifact")
        ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1], s=4, alpha=0.35, color="tab:blue", label="star")
        xlim = (-2.5, 4)
        key = "2 features, stars vs hot pixels" if "hot" in title else "2 features, all candidates"
        p = percs[key]
        boundary_line(ax, p.w, p.b, xlim, color="k", lw=2, label="perceptron (final weights)")
        if "hot" in title:
            m = LogisticRegressionGD(lam=LAM).fit(Z["train"][m_hp["train"]][:, TWO],
                                                  y["train"][m_hp["train"]])
        else:
            m = lrs["2 features"][0]
        boundary_line(ax, m.theta[1:], m.theta[0], xlim, color="tab:green", lw=2, ls="--",
                      label="logistic p = 0.5")
        for lev, ls in ((0.1, ":"), (0.9, ":")):
            c = np.log(lev / (1 - lev))
            boundary_line(ax, m.theta[1:], m.theta[0] - c, xlim, color="tab:green", lw=1, ls=ls)
        ax.set_xlim(xlim)
        ax.set_ylim(-2.5, 4.5)
        ax.set_xlabel("log width (standardized)")
        ax.set_ylabel("roundness (standardized)")
        ax.set_title(f"{title} (test set); dotted = p 0.1 / 0.9")
        ax.legend(fontsize=8, markerscale=3, loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_boundaries_2d.png", dpi=140)
    plt.close(fig)

    # confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, yhat) in zip(axes, (("perceptron", percs["5 features, all candidates"].predict(Z["test"])),
                                       ("logistic regression", lrs["5 features"][0].predict(Z["test"])))):
        C = confusion(y["test"], yhat)
        ax.imshow(C, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, C[i, j], ha="center", va="center", fontsize=13,
                        color="w" if C[i, j] > C.max() / 2 else "k")
        ax.set_xticks([0, 1], ["pred artifact", "pred star"])
        ax.set_yticks([0, 1], ["true artifact", "true star"])
        ax.set_title(f"{name}: acc {np.mean(yhat == y['test']):.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_confusion.png", dpi=140)
    plt.close(fig)
    for f in ("exp_b_learning_curves.png", "exp_b_boundaries_2d.png", "exp_b_confusion.png"):
        print(f"  saved {OUT / f}")


if __name__ == "__main__":
    main()
