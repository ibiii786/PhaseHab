r"""
06_compare.py - Statistical comparison of the CV models (config/analysis_plan.md, section 8).

Input: outputs/models/oof_predictions.csv (from 05_train.py; development set only, test never used).
Each patient's out-of-fold probability is averaged over the 5 repeats (plan section 8), then:

1. Metrics per model x classifier x cohort (pooled / I-SPY1 / I-SPY2):
   AUROC with DeLong 95% CI and stratified bootstrap 95% CI (2,000 resamples), AUPRC and Brier with bootstrap CIs,
   calibration-in-the-large and calibration slope.
2. Threshold metrics: sensitivity, specificity, F1 at the Youden threshold. For each fold the threshold is chosen
   on the OOF predictions of the OTHER 4 folds of the same repeat (= that fold's training patients), then applied
   to the fold. Mean +- SD over the 5 repeats.
3. DeLong tests. PRIMARY family (lr, pooled), Holm over exactly 4 tests:
   early vs pre, early vs late, late vs pre, fused vs best single phase.
   Same 4 tests for xgb (secondary) and per cohort (exploratory), each with its own Holm correction.
   Extra, unadjusted, descriptive: each phase vs clinical, clinical + best phase vs clinical,
   clinical + arm vs clinical (I-SPY2 only).
4. Figures (outputs/compare/figures/): ROC curves, calibration curves, AUROC forest plot, phase-difference plot.

Run from code\ :  .venv\Scripts\python.exe src\06_compare.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, roc_curve)  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import paths as P  # noqa: E402
import stats as S  # noqa: E402

MOD = P.OUT / "models"
OUT = P.OUT / "compare"
FIG = OUT / "figures"
N_BOOT = 2000
COHORTS = ["pooled", "spy1", "spy2"]
MODELS = ["clinical", "pre", "early", "late", "fused", "clinical_best", "clinical_arm"]
LABEL = {"clinical": "Clinical", "pre": "Pre-contrast", "early": "Early post", "late": "Late post",
         "fused": "Fused (3 phases + shape)", "clinical_best": "Clinical + best phase",
         "clinical_arm": "Clinical + arm (I-SPY2)"}
# categorical palette, fixed order (validated with the dataviz validator; legend + line style as 2nd channel)
COLOUR = {"pre": "#2a78d6", "early": "#eb6834", "late": "#1baf7a", "fused": "#eda100", "clinical": "#e87ba4"}
STYLE = {"pre": "-", "early": "--", "late": "-.", "fused": ":", "clinical": (0, (5, 1, 1, 1))}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def cohort_rows(df, cohort):
    return df if cohort == "pooled" else df[df["dataset"] == cohort]


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- 1. metrics
def model_metrics(m, c, cohort, g, boot_idx):
    y, p = g["pCR"].to_numpy(), g["prob"].to_numpy()
    auc, lo, hi = S.delong_ci(y, p)
    row = {"model": m, "classifier": c, "cohort": cohort, "n": len(g), "n_pCR": int(y.sum()),
           "AUROC": auc, "AUROC_delong_lo": lo, "AUROC_delong_hi": hi,
           "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
           "Brier_no_skill": y.mean() * (1 - y.mean())}
    row.update(S.boot_ci(y, p, boot_idx))
    row.update(S.calibration(y, p))
    return row


# ----------------------------------------------------------------------------- 2. thresholds
def threshold_metrics(oof):
    rows = []
    for (m, c, rep), g in oof.groupby(["model", "classifier", "repeat"]):
        pred = pd.Series(0, index=g.index)
        for f in g["fold"].unique():
            other = g[g["fold"] != f]
            thr = S.youden_threshold(other["pCR"], other["prob"])
            sel = g["fold"] == f
            pred[sel] = (g.loc[sel, "prob"] >= thr).astype(int)
        g = g.assign(pred=pred)
        for cohort in COHORTS:
            gg = cohort_rows(g, cohort)
            if gg.empty:
                continue
            y, yhat = gg["pCR"], gg["pred"]
            rows.append({"model": m, "classifier": c, "repeat": rep, "cohort": cohort,
                         "sensitivity": ((yhat == 1) & (y == 1)).sum() / (y == 1).sum(),
                         "specificity": ((yhat == 0) & (y == 0)).sum() / (y == 0).sum(),
                         "F1": f1_score(y, yhat, zero_division=0)})
    per_rep = pd.DataFrame(rows)
    return per_rep.groupby(["model", "classifier", "cohort"]).agg(
        sens_mean=("sensitivity", "mean"), sens_sd=("sensitivity", "std"),
        spec_mean=("specificity", "mean"), spec_sd=("specificity", "std"),
        F1_mean=("F1", "mean"), F1_sd=("F1", "std")).reset_index()


# ----------------------------------------------------------------------------- 3. tests
def run_tests(avg, best):
    primary = [("early", "pre"), ("early", "late"), ("late", "pre"), ("fused", best)]
    extra = [("pre", "clinical"), ("early", "clinical"), ("late", "clinical"), ("fused", "clinical"),
             ("clinical_best", "clinical")]
    rows = []
    for c in ["lr", "xgb"]:
        for cohort in COHORTS:
            fam = "primary" if (c == "lr" and cohort == "pooled") else (
                "secondary (xgb)" if cohort == "pooled" else "exploratory (cohort)")
            block = []
            for a, b in primary + extra:
                ga = cohort_rows(avg[(avg.model == a) & (avg.classifier == c)], cohort).set_index("pid")
                gb = cohort_rows(avg[(avg.model == b) & (avg.classifier == c)], cohort).set_index("pid")
                ga, gb = ga.align(gb, join="inner", axis=0)
                t = S.delong_test(ga["pCR"].to_numpy(), ga["prob"].to_numpy(), gb["prob"].to_numpy())
                block.append({"classifier": c, "cohort": cohort, "comparison": f"{a} vs {b}",
                              "family": fam if (a, b) in primary else "extra (unadjusted)",
                              "n": len(ga), **t})
            ps = [r["p"] for r in block if r["family"] != "extra (unadjusted)"]
            adj = iter(S.holm(ps))
            for r in block:
                r["p_holm"] = next(adj) if r["family"] != "extra (unadjusted)" else np.nan
            rows += block
    # clinical + arm vs clinical, I-SPY2 only, lr
    ga = avg[(avg.model == "clinical_arm") & (avg.classifier == "lr")].set_index("pid")
    gb = avg[(avg.model == "clinical") & (avg.classifier == "lr")].set_index("pid").loc[ga.index]
    t = S.delong_test(ga["pCR"].to_numpy(), ga["prob"].to_numpy(), gb["prob"].to_numpy())
    rows.append({"classifier": "lr", "cohort": "spy2", "comparison": "clinical_arm vs clinical",
                 "family": "sensitivity (unadjusted)", "n": len(ga), **t, "p_holm": np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. figures
def fig_roc(avg, clf, cohort, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    for m in ["clinical", "pre", "early", "late", "fused"]:
        g = cohort_rows(avg[(avg.model == m) & (avg.classifier == clf)], cohort)
        fpr, tpr, _ = roc_curve(g["pCR"], g["prob"])
        auc = S.fast_auc(g["pCR"].to_numpy(), g["prob"].to_numpy())
        ax.plot(fpr, tpr, color=COLOUR[m], linestyle=STYLE[m], linewidth=2, label=f"{LABEL[m]}  {auc:.3f}")
    style_axes(ax)
    ax.set_xlabel("1 - specificity", color=INK)
    ax.set_ylabel("Sensitivity", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    name = {"pooled": "I-SPY1 + I-SPY2", "spy1": "I-SPY1", "spy2": "I-SPY2"}[cohort]
    ax.set_title(f"ROC, cross-validation ({name}, {clf.upper()})", color=INK, fontsize=11, loc="left")
    leg = ax.legend(title="Model  AUROC", loc="lower right", fontsize=8.5, title_fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_calibration(avg, clf, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (2, 3)), label="Perfect calibration")
    for m in ["clinical", "pre", "early", "late", "fused"]:
        g = avg[(avg.model == m) & (avg.classifier == clf)]
        obs, pred = calibration_curve(g["pCR"], g["prob"], n_bins=10, strategy="quantile")
        ax.plot(pred, obs, color=COLOUR[m], linestyle=STYLE[m], linewidth=2, marker="o", markersize=4,
                label=LABEL[m])
    style_axes(ax)
    ax.axhline(avg["pCR"].mean(), color=MUTED, linewidth=0.8)
    ax.text(0.02, avg["pCR"].mean() + 0.015, "observed pCR rate", color=MUTED, fontsize=8)
    ax.set_xlabel("Mean predicted probability (10 equal-size groups)", color=INK)
    ax.set_ylabel("Observed pCR rate", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration, cross-validation (pooled, {clf.upper()})", color=INK, fontsize=11, loc="left")
    leg = ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_forest(met, path):
    models = ["clinical", "pre", "early", "late", "fused", "clinical_best"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        for j, clf in enumerate(["lr", "xgb"]):
            g = met[(met.cohort == cohort) & (met.classifier == clf)].set_index("model")
            for i, m in enumerate(models):
                if m not in g.index:
                    continue
                y = i + (-0.15 if clf == "lr" else 0.15)
                r = g.loc[m]
                col = INK if clf == "lr" else MUTED
                ax.plot([r["AUROC_lo"], r["AUROC_hi"]], [y, y], color=col, linewidth=1.5)
                ax.plot(r["AUROC"], y, marker="o" if clf == "lr" else "s", markersize=6, color=col,
                        markerfacecolor=col if clf == "lr" else "white", linestyle="none",
                        label=("LR (primary)" if clf == "lr" else "XGBoost") if i == 0 else None)
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
        style_axes(ax)
        ax.set_xlim(0.3, 0.9)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([LABEL[m] for m in models])
        ax.invert_yaxis()
        n = int(met[(met.cohort == cohort) & (met.model == "clinical")]["n"].iloc[0])
        ax.set_title({"pooled": "Pooled", "spy1": "I-SPY1", "spy2": "I-SPY2"}[cohort] + f" (n = {n})",
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("AUROC (95% bootstrap CI)", color=INK)
    leg = axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_differences(tests, path):
    g = tests[(tests.family.isin(["primary", "secondary (xgb)"]))].copy()
    comps = g[g.classifier == "lr"]["comparison"].tolist()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, comp in enumerate(comps):
        for clf, dy in [("lr", -0.13), ("xgb", 0.13)]:
            r = g[(g.comparison == comp) & (g.classifier == clf)].iloc[0]
            col = INK if clf == "lr" else MUTED
            ax.plot([r["diff_lo"], r["diff_hi"]], [i + dy, i + dy], color=col, linewidth=1.5)
            ax.plot(r["diff"], i + dy, marker="o" if clf == "lr" else "s", markersize=6, color=col,
                    markerfacecolor=col if clf == "lr" else "white", linestyle="none",
                    label=("LR (primary)" if clf == "lr" else "XGBoost") if i == 0 else None)
            ax.text(0.105, i + dy, f"Holm p = {r['p_holm']:.2f}", va="center", fontsize=8, color=col)
    ax.axvline(0, color=MUTED, linewidth=1)
    style_axes(ax)
    ax.set_xlim(-0.1, 0.1)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([c.replace(" vs ", " minus ") for c in comps])
    ax.invert_yaxis()
    ax.set_xlabel("Difference in AUROC (DeLong 95% CI), pooled cross-validation", color=INK)
    ax.set_title("Phase comparison (pre-registered 4 tests)", color=INK, fontsize=11, loc="left")
    leg = ax.legend(loc="lower left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(MOD / "oof_predictions.csv")
    cvm = pd.read_csv(MOD / "cv_metrics.csv")
    ph = cvm[(cvm.cohort == "pooled") & (cvm.classifier == "lr") & cvm.model.isin(["pre", "early", "late"])]
    best = ph.sort_values("AUROC_mean", ascending=False)["model"].iloc[0]
    print(f"OOF rows: {len(oof):,}; patients: {oof.pid.nunique()} (development set only). Best phase (CV): {best}")

    avg = oof.groupby(["model", "classifier", "pid", "dataset"], as_index=False).agg(
        pCR=("pCR", "first"), prob=("prob", "mean"))
    avg.to_csv(OUT / "oof_repeat_averaged.csv", index=False)

    # 1. metrics with CIs (same bootstrap samples for every model within a cohort, so CIs are comparable)
    ref = avg[(avg.model == "clinical") & (avg.classifier == "lr")].sort_values("pid")
    tasks = []
    for cohort in COHORTS + ["spy2_arm"]:
        coh = "spy2" if cohort == "spy2_arm" else cohort
        base = cohort_rows(ref, coh).reset_index(drop=True)
        idx = S.boot_indices(base["pCR"].to_numpy(), N_BOOT, P.SEED)
        for (m, c), g in avg.groupby(["model", "classifier"]):
            if (m == "clinical_arm") != (cohort == "spy2_arm"):
                continue
            g = cohort_rows(g, coh).set_index("pid").loc[base["pid"]].reset_index()
            tasks.append(delayed(model_metrics)(m, c, coh, g, idx))
    met = pd.DataFrame(Parallel(n_jobs=4)(tasks))
    rep = cvm.rename(columns={"AUROC_mean": "AUROC_rep_mean", "AUROC_sd": "AUROC_rep_sd"})[
        ["model", "classifier", "cohort", "AUROC_rep_mean", "AUROC_rep_sd"]]
    thr = threshold_metrics(oof)
    met = met.merge(rep, on=["model", "classifier", "cohort"], how="left") \
             .merge(thr, on=["model", "classifier", "cohort"], how="left")
    met["order"] = met["model"].map({m: i for i, m in enumerate(MODELS)})
    met = met.sort_values(["cohort", "classifier", "order"]).drop(columns="order")
    met.round(4).to_csv(OUT / "metrics_table.csv", index=False)

    # 3. tests
    tests = run_tests(avg, best)
    tests.round(4).to_csv(OUT / "delong_tests.csv", index=False)

    # 4. figures
    for clf in ["lr", "xgb"]:
        fig_roc(avg, clf, "pooled", FIG / f"roc_pooled_{clf}.png")
        fig_calibration(avg, clf, FIG / f"calibration_pooled_{clf}.png")
    for cohort in ["spy1", "spy2"]:
        fig_roc(avg, "lr", cohort, FIG / f"roc_{cohort}_lr.png")
    fig_forest(met, FIG / "auroc_forest.png")
    fig_differences(tests, FIG / "phase_differences.png")

    # console summary
    cols = ["model", "classifier", "AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "Brier", "Brier_no_skill",
            "slope", "citl", "sens_mean", "spec_mean", "F1_mean"]
    print("\nPOOLED (repeat-averaged OOF)\n" + met[met.cohort == "pooled"][cols].round(3).to_string(index=False))
    print("\nBY COHORT, LR\n" + met[(met.cohort != "pooled") & (met.classifier == "lr")][
        ["model", "cohort", "n", "n_pCR", "AUROC", "AUROC_lo", "AUROC_hi"]].round(3).to_string(index=False))
    show = ["classifier", "cohort", "comparison", "family", "auc_a", "auc_b", "diff", "diff_lo", "diff_hi", "p",
            "p_holm"]
    print("\nDELONG TESTS\n" + tests[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
