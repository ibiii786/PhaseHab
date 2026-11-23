import argparse
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import cv
import paths as P
from modelling import fit_pipe, kept_names, load_data, make_pipeline

OUT = FIG = None
GROUPINGS = ["phase", "habitat", "phase_habitat", "class"]
COLOUR = {"pre": "#2a78d6", "early": "#eb6834", "late": "#1baf7a", "shape": "#eda100", "clinical": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
PHASE_NAME = {"pre": "Pre-contrast", "early": "Early post", "late": "Late post", "shape": "Shape",
              "clinical": "Clinical"}


def group_of(name, grouping):
    tok = name.split("_")
    if tok[0] in ("pre", "early", "late"):
        phase, hab, cls = tok[0], tok[1], tok[2]
    elif tok[0] == "shape":
        phase = hab = cls = "shape"
    else:
        phase = hab = cls = "clinical"
    ph_hab = phase if phase in ("shape", "clinical") else f"{phase}_{hab}"
    return {"phase": phase, "habitat": hab, "phase_habitat": ph_hab, "class": cls}[grouping]


def explain(pipe, X_train, X_explain):
    front = pipe[:-1]
    clf = pipe.named_steps["clf"]
    names = kept_names(pipe, X_train.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zt = np.asarray(front.transform(X_train), float)
        Ze = np.asarray(front.transform(X_explain), float)
        if hasattr(clf, "coef_"):
            ex = shap.LinearExplainer(clf, shap.maskers.Independent(Zt, max_samples=len(Zt)))
            sv = ex.shap_values(Ze)
        else:
            sv = shap.TreeExplainer(clf).shap_values(Ze)
    return np.asarray(sv), names, Ze


def group_importance(sv, names):
    rows = []
    for grouping in GROUPINGS:
        g = pd.Series([group_of(n, grouping) for n in names])
        imp = {grp: float(np.abs(sv[:, (g == grp).to_numpy()].sum(1)).mean()) for grp in g.unique()}
        tot = sum(imp.values())
        for grp, v in imp.items():
            rows.append({"grouping": grouping, "group": grp, "importance": v,
                         "share": v / tot if tot > 0 else np.nan, "n_features": int((g == grp).sum())})
    return rows


def fold_job(model, clf, X, y, rep, fold, tr, va):
    pipe = fit_pipe(make_pipeline("big", clf), X.iloc[tr], y[tr])
    sv, names, _ = explain(pipe, X.iloc[tr], X.iloc[va])
    return [{"model": model, "classifier": clf, "repeat": rep, "fold": fold, **r}
            for r in group_importance(sv, names)]


def style_axes(ax, grid_axis="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_phase_shares(summ, path):
    groups = ["pre", "early", "late", "shape"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, clf in zip(axes, ["lr", "xgb"]):
        s = summ[(summ.model == "fused") & (summ.classifier == clf) & (summ.grouping == "phase")].set_index("group")
        for i, grp in enumerate(groups):
            m = 100 * s.loc[grp, "share_mean"] if grp in s.index else 0
            sd = 100 * s.loc[grp, "share_sd"] if grp in s.index else 0
            ax.barh(i, m, height=0.6, color=COLOUR[grp], edgecolor="white", linewidth=2)
            ax.errorbar(m, i, xerr=sd, color=INK, capsize=3, linewidth=1)
            ax.text(m + sd + 1.5, i, f"{m:.0f}%", va="center", fontsize=9, color=INK)
        style_axes(ax)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([PHASE_NAME[g] for g in groups])
        ax.invert_yaxis()
        ax.set_xlim(0, 75)
        ax.set_xlabel("Share of SHAP importance (%), mean ± SD over 25 folds", color=INK, fontsize=9)
        ax.set_title({"lr": "Fused model, LR (primary)", "xgb": "Fused model, XGBoost"}[clf], color=INK,
                     fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_habitat_shares(summ, k, path):
    habs = [f"h{h}" for h in range(1, k + 1)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, ph in zip(axes, ["pre", "early", "late"]):
        s = summ[(summ.model == ph) & (summ.classifier == "lr") & (summ.grouping == "habitat")].set_index("group")
        for i, h in enumerate(habs):
            m = 100 * s.loc[h, "share_mean"] if h in s.index else 0
            sd = 100 * s.loc[h, "share_sd"] if h in s.index else 0
            ax.bar(i, m, width=0.6, color=COLOUR[ph], edgecolor="white", linewidth=2)
            ax.errorbar(i, m, yerr=sd, color=INK, capsize=3, linewidth=1)
            ax.text(i, m + sd + 2, f"{m:.0f}%", ha="center", fontsize=9, color=INK)
        style_axes(ax, "y")
        ax.set_xticks(range(len(habs)))
        ax.set_xticklabels(["h1 low", "h2 medium", "h3 high"][:k] if k == 3 else habs)
        ax.set_ylim(0, 100)
        ax.set_title(f"{PHASE_NAME[ph]} model (LR)", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Share of SHAP importance (%)\nmean ± SD over 25 folds", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heatmap(summ, k, path):
    phases = ["pre", "early", "late"]
    habs = [f"h{h}" for h in range(1, k + 1)]
    s = summ[(summ.model == "fused") & (summ.classifier == "lr") & (summ.grouping == "phase_habitat")] \
        .set_index("group")["share_mean"]
    M = np.array([[100 * s.get(f"{p}_{h}", 0.0) for h in habs] for p in phases])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"]), vmin=0, vmax=max(M.max(), 1))
    for i in range(len(phases)):
        for j in range(len(habs)):
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > 0.6 * M.max() else INK)
    ax.set_xticks(range(len(habs)))
    ax.set_xticklabels(["h1 low", "h2 medium", "h3 high"][:k] if k == 3 else habs)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([PHASE_NAME[p] for p in phases])
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sh = 100 * s.get("shape", 0.0)
    ax.set_title(f"Fused LR: share by phase × habitat\n(25-fold mean; 9 cells + shape {sh:.0f}% = 100%)",
                 color=INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of SHAP importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_beeswarm(sv, Z, names, title, path):
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=names, max_display=15, show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.suptitle(title, color=INK, fontsize=10)
    fig.axes[0].set_xlabel("SHAP value (log-odds of pCR)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    global OUT, FIG
    OUT = P.OUT / "shap" / f"k{args.k}"
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mdir = P.OUT / "models" / f"k{args.k}"

    dev, X = load_data(args.k)
    y = dev["pCR"].astype(int).to_numpy()
    allX = pd.concat([X["clinical"], X["fused"]], axis=1)
    print(f"Development patients: {len(dev)}; test set not loaded.")

    final_rows, top_rows = [], []
    for model, clf in [("pre", "lr"), ("early", "lr"), ("late", "lr"), ("fused", "lr"), ("clinical_best", "lr"),
                       ("fused", "xgb")]:
        d = joblib.load(mdir / f"{model}_k{args.k}_{clf}.joblib")
        Xm = allX[d["features"]]
        sv, names, Z = explain(d["pipeline"], Xm, Xm)
        pd.DataFrame(sv, columns=names).assign(pid=dev["pid"].to_numpy()) \
            .to_csv(OUT / f"final_shap_{model}_{clf}.csv", index=False)
        for r in group_importance(sv, names):
            final_rows.append({"model": model, "classifier": clf, **r})
        freq = {}
        fpath = mdir / f"selected_features_{model}_k{args.k}.csv"
        if clf == "lr" and fpath.exists():
            freq = pd.read_csv(fpath).set_index("feature")["fraction"].to_dict()
        mabs = np.abs(sv).mean(0)
        for rank, j in enumerate(np.argsort(-mabs)[:20], 1):
            rho = spearmanr(Z[:, j], sv[:, j]).statistic if np.std(Z[:, j]) > 0 else np.nan
            top_rows.append({"model": model, "classifier": clf, "rank": rank, "feature": names[j],
                             "mean_abs_shap": mabs[j],
                             "direction": "higher value -> more likely pCR" if rho > 0 else
                             "higher value -> less likely pCR",
                             "lasso_fold_fraction": freq.get(names[j], np.nan)})
        if model == "fused":
            fig_beeswarm(sv, Z, names, f"Fused model ({clf.upper()}), final fit, development set",
                         FIG / f"beeswarm_fused_{clf}.png")
        print(f"final {model}/{clf}: {len(names)} features explained")
    final = pd.DataFrame(final_rows)
    final.round(4).to_csv(OUT / "group_importance_final.csv", index=False)
    pd.DataFrame(top_rows).round(4).to_csv(OUT / "top_features_final.csv", index=False)

    folds = cv.dev_folds(dev)
    tasks = [delayed(fold_job)(m, c, X[m], y, rep, f, tr, va)
             for m, c in [("fused", "lr"), ("fused", "xgb"), ("pre", "lr"), ("early", "lr"), ("late", "lr")]
             for rep, f, tr, va in folds]
    rows = [r for res in Parallel(n_jobs=args.jobs, verbose=5)(tasks) for r in res]
    per_fold = pd.DataFrame(rows)
    full = []
    for (m, c, g), d in per_fold.groupby(["model", "classifier", "grouping"]):
        piv = d.pivot_table(index=["repeat", "fold"], columns="group", values="share").fillna(0.0)
        for grp in piv.columns:
            full.append({"model": m, "classifier": c, "grouping": g, "group": grp,
                         "share_mean": piv[grp].mean(), "share_sd": piv[grp].std(),
                         "share_min": piv[grp].min(), "share_max": piv[grp].max(),
                         "folds_with_zero": int((piv[grp] == 0).sum()), "n_folds": len(piv)})
    summ = pd.DataFrame(full).sort_values(["model", "classifier", "grouping", "share_mean"],
                                          ascending=[True, True, True, False])
    per_fold.round(4).to_csv(OUT / "group_importance_folds.csv", index=False)
    summ.round(4).to_csv(OUT / "group_importance_summary.csv", index=False)

    fig_phase_shares(summ, FIG / "fused_share_by_phase.png")
    fig_habitat_shares(summ, args.k, FIG / "phase_models_share_by_habitat.png")
    fig_heatmap(summ, args.k, FIG / "fused_phase_x_habitat.png")

    def show(model, clf, grouping):
        s = summ[(summ.model == model) & (summ.classifier == clf) & (summ.grouping == grouping)]
        f = final[(final.model == model) & (final.classifier == clf) & (final.grouping == grouping)] \
            .set_index("group")["share"]
        txt = ", ".join(f"{r.group} {100*r.share_mean:.0f}±{100*r.share_sd:.0f}% (final {100*f.get(r.group, 0):.0f}%)"
                        for r in s.itertuples())
        print(f"  {model}/{clf} by {grouping}: {txt}")

    print("\nSHARE OF SHAP IMPORTANCE (folds mean ± SD; final model in brackets)")
    for m, c in [("fused", "lr"), ("fused", "xgb")]:
        show(m, c, "phase")
        show(m, c, "habitat")
    for m in ["pre", "early", "late"]:
        show(m, "lr", "habitat")
    fb = final[(final.model == "clinical_best") & (final.grouping == "phase")].set_index("group")["share"]
    print("  clinical_best/lr (final) by block: " + ", ".join(f"{g} {100*v:.0f}%" for g, v in fb.items()))
    tf = pd.DataFrame(top_rows)
    print("\nTOP 10 FEATURES, fused LR (final)\n" + tf[(tf.model == "fused") & (tf.classifier == "lr")].head(10)[
        ["rank", "feature", "mean_abs_shap", "direction", "lasso_fold_fraction"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
