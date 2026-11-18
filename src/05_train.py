r"""
05_train.py - Train and cross-validate the models (config/analysis_plan.md, sections 2, 7 and 8).

Development set = train + val (about 1,010 patients). The TEST SET IS NEVER LOADED HERE.
CV = repeated stratified 5-fold, 5 repeats, strata pCR x subtype, seed 42 (src/cv.py, same folds as 03).

Models (feature blocks from outputs/features/k{k}/):
  clinical  HR, HER2, age, cohort (I-SPY2 = 1)
  pre       pre_h*  features                 (no shape: shape is the same for every phase)
  early     early_h* features
  late      late_h*  features
  fused     pre + early + late + shape
  clinical_best   clinical + the best single phase (highest mean CV AUROC with the primary classifier)
  clinical_arm    sensitivity analysis, I-SPY2 only: clinical + grouped treatment arm

Classifiers (everything learned is inside an sklearn Pipeline, so it is fit on the training folds only):
  lr   (PRIMARY) impute (median) -> rank-to-normal scaling -> drop near-constant -> drop |r| > 0.95
       -> LASSO selection (L1 logistic, C chosen by inner 5-fold CV AUROC) -> L2 logistic (C by inner CV).
       class_weight = balanced.
  xgb  (secondary) impute -> drop near-constant -> drop |r| > 0.95 -> XGBoost (fixed, conservative settings,
       balanced sample weights). No LASSO step: trees do their own selection.
  The clinical model has only 4 inputs, so its lr pipeline skips the filters and LASSO.

Outputs (outputs/models/k{k}/):
  oof_predictions.csv     one row per patient x model x classifier x repeat (out-of-fold probability)
  cv_metrics_by_repeat.csv, cv_metrics.csv (AUROC / AUPRC / Brier: mean +- SD over repeats,
                          plus AUROC of the repeat-averaged OOF probability, pooled / I-SPY1 / I-SPY2)
  cv_differences.csv      paired AUROC differences between models, per repeat (descriptive; tests are in 06)
  selected_features_<model>_k{k}.csv   how often LASSO picked each feature over the 25 folds
  <model>_k{k}_<classifier>.joblib     final pipeline refit on the full development set
  DONE.txt                written last, so the overnight runner knows this k is finished

Run from code\ :  .venv\Scripts\python.exe src\05_train.py [--k 3] [--jobs 4]
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import cv  # noqa: E402
import paths as P  # noqa: E402
from modelling import fit_pipe, kept_names, load_data, make_pipeline  # noqa: E402


# ----------------------------------------------------------------------------- CV
def run_fold(model, clf, kind, X, y, rep, fold, tr, va):
    pipe = fit_pipe(make_pipeline(kind, clf), X.iloc[tr], y[tr])
    prob = pipe.predict_proba(X.iloc[va])[:, 1]
    sel = kept_names(pipe, X.columns) if clf == "lr" and kind != "small" else []
    return {"model": model, "classifier": clf, "repeat": rep, "fold": fold, "idx": va, "prob": prob,
            "selected": sel, "n_selected": len(sel) if sel else X.shape[1]}


def metrics(y, p):
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    out = P.OUT / "models" / f"k{args.k}"
    out.mkdir(parents=True, exist_ok=True)

    dev, X = load_data(args.k)
    y = dev["pCR"].astype(int).to_numpy()
    folds = cv.dev_folds(dev)
    is_spy2 = (dev["dataset"] == "spy2").to_numpy()
    print(f"Development patients: {len(dev)} (pCR rate {y.mean():.3f}); test set not loaded. "
          f"{len(folds)} folds. Features: " + ", ".join(f"{m} {x.shape[1]}" for m, x in X.items()))

    # job list: (model, classifier, kind, row subset)
    jobs = [("clinical", c, "small", None) for c in ["lr", "xgb"]]
    jobs += [(m, c, "big", None) for m in ["pre", "early", "late", "fused"] for c in ["lr", "xgb"]]
    jobs += [("clinical_arm", "lr", "small", is_spy2)]

    def run(job_list):
        tasks = []
        for model, clf, kind, sub in job_list:
            for rep, fold, tr, va in folds:
                if sub is not None:
                    tr, va = tr[sub[tr]], va[sub[va]]
                tasks.append(delayed(run_fold)(model, clf, kind, X[model], y, rep, fold, tr, va))
        return Parallel(n_jobs=args.jobs, verbose=5)(tasks)

    res = run(jobs)

    # clinical + best single phase (chosen on CV AUROC of the primary classifier, never on test)
    def mean_auc(model):
        r = [x for x in res if x["model"] == model and x["classifier"] == "lr"]
        aucs = []
        for rep in range(cv.N_REPEATS):
            rr = [x for x in r if x["repeat"] == rep]
            idx = np.concatenate([x["idx"] for x in rr])
            aucs.append(roc_auc_score(y[idx], np.concatenate([x["prob"] for x in rr])))
        return np.mean(aucs)

    phase_auc = {m: mean_auc(m) for m in ["pre", "early", "late"]}
    best = max(phase_auc, key=phase_auc.get)
    print("Mean CV AUROC (lr) by phase:", {m: round(a, 3) for m, a in phase_auc.items()}, "-> best:", best)
    X["clinical_best"] = pd.concat([X["clinical"], X[best]], axis=1)
    res += run([("clinical_best", c, "big", None) for c in ["lr", "xgb"]])

    # ---- out-of-fold predictions
    rows = []
    for r in res:
        d = dev.iloc[r["idx"]][["pid", "dataset", "pCR", "subtype"]].copy()
        d["model"], d["classifier"], d["repeat"], d["fold"], d["prob"] = \
            r["model"], r["classifier"], r["repeat"], r["fold"], r["prob"]
        rows.append(d)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)

    # ---- metrics per repeat, and of the repeat-averaged probability
    per_rep = []
    for (m, c, rep), g in oof.groupby(["model", "classifier", "repeat"]):
        for cohort, gg in [("pooled", g), ("spy1", g[g.dataset == "spy1"]), ("spy2", g[g.dataset == "spy2"])]:
            if gg["pCR"].nunique() == 2:
                per_rep.append({"model": m, "classifier": c, "repeat": rep, "cohort": cohort, "n": len(gg),
                                **metrics(gg["pCR"], gg["prob"])})
    per_rep = pd.DataFrame(per_rep)
    per_rep.to_csv(out / "cv_metrics_by_repeat.csv", index=False)

    avg = oof.groupby(["model", "classifier", "pid", "dataset"], as_index=False).agg(pCR=("pCR", "first"),
                                                                                      prob=("prob", "mean"))
    summ = []
    for (m, c, cohort), g in per_rep.groupby(["model", "classifier", "cohort"]):
        a = avg[(avg.model == m) & (avg.classifier == c)]
        if cohort != "pooled":
            a = a[a.dataset == cohort]
        sel = [r["n_selected"] for r in res if r["model"] == m and r["classifier"] == c]
        summ.append({"model": m, "classifier": c, "cohort": cohort, "n": int(g["n"].iloc[0]),
                     "AUROC_mean": g["AUROC"].mean(), "AUROC_sd": g["AUROC"].std(),
                     "AUPRC_mean": g["AUPRC"].mean(), "AUPRC_sd": g["AUPRC"].std(),
                     "Brier_mean": g["Brier"].mean(), "Brier_sd": g["Brier"].std(),
                     "AUROC_avg_oof": roc_auc_score(a["pCR"], a["prob"]),
                     "median_features_used": float(np.median(sel))})
    order = {m: i for i, m in enumerate(["clinical", "pre", "early", "late", "fused", "clinical_best",
                                         "clinical_arm"])}
    summ = pd.DataFrame(summ).sort_values(["cohort", "classifier", "model"],
                                          key=lambda s: s.map(order) if s.name == "model" else s)
    summ.round(4).to_csv(out / "cv_metrics.csv", index=False)

    # ---- paired differences per repeat (descriptive; DeLong + Holm in 06_compare.py)
    pairs = [("early", "pre"), ("early", "late"), ("late", "pre"), ("fused", best), ("pre", "clinical"),
             ("early", "clinical"), ("late", "clinical"), ("clinical_best", "clinical")]
    pooled = per_rep[per_rep.cohort == "pooled"].set_index(["model", "classifier", "repeat"])["AUROC"]
    diffs = []
    for c in ["lr", "xgb"]:
        for a, b in pairs:
            d = np.array([pooled[(a, c, r)] - pooled[(b, c, r)] for r in range(cv.N_REPEATS)])
            diffs.append({"classifier": c, "comparison": f"{a} - {b}", "AUROC_diff_mean": d.mean(),
                          "AUROC_diff_sd": d.std(ddof=1), "repeats_a_better": int((d > 0).sum())})
    diffs = pd.DataFrame(diffs).round(4)
    diffs.to_csv(out / "cv_differences.csv", index=False)

    # ---- how often LASSO picked each feature
    for m in ["pre", "early", "late", "fused", "clinical_best"]:
        sel = [r["selected"] for r in res if r["model"] == m and r["classifier"] == "lr"]
        s = pd.Series([f for fs in sel for f in fs]).value_counts().rename("folds_selected")
        s = s.to_frame()
        s["fraction"] = (s["folds_selected"] / len(sel)).round(3)
        s.rename_axis("feature").to_csv(out / f"selected_features_{m}_k{args.k}.csv")

    # ---- refit every model on the full development set and save it
    final = [("clinical", c, "small") for c in ["lr", "xgb"]]
    final += [(m, c, "big") for m in ["pre", "early", "late", "fused", "clinical_best"] for c in ["lr", "xgb"]]
    final += [("clinical_arm", "lr", "small")]

    def refit(model, clf, kind):
        rows_ = is_spy2 if model == "clinical_arm" else np.ones(len(dev), bool)
        pipe = fit_pipe(make_pipeline(kind, clf), X[model][rows_], y[rows_])
        return model, clf, pipe

    for model, clf, pipe in Parallel(n_jobs=args.jobs)(delayed(refit)(*f) for f in final):
        joblib.dump({"pipeline": pipe, "features": list(X[model].columns), "model": model, "classifier": clf,
                     "k": args.k, "best_phase": best, "selected": kept_names(pipe, X[model].columns),
                     "trained_on": "train + val (development set)"},
                    out / f"{model}_k{args.k}_{clf}.joblib")

    show = summ[summ.cohort == "pooled"][["model", "classifier", "AUROC_mean", "AUROC_sd", "AUROC_avg_oof",
                                          "AUPRC_mean", "Brier_mean", "median_features_used"]]
    print("\nPOOLED CV RESULTS\n" + show.round(3).to_string(index=False))
    coh = summ[summ.cohort != "pooled"].pivot_table(index=["model", "classifier"], columns="cohort",
                                                    values="AUROC_mean")
    print("\nAUROC by cohort\n" + coh.round(3).to_string())
    print("\nPAIRED DIFFERENCES (pooled AUROC, per repeat)\n" + diffs.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    (out / "DONE.txt").write_text(f"05_train.py finished, k = {args.k}, {time.strftime('%Y-%m-%d %H:%M')}\n")


if __name__ == "__main__":
    main()
