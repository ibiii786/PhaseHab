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

Outputs (outputs/models/):
  oof_predictions.csv     one row per patient x model x classifier x repeat (out-of-fold probability)
  cv_metrics_by_repeat.csv, cv_metrics.csv (AUROC / AUPRC / Brier: mean +- SD over repeats,
                          plus AUROC of the repeat-averaged OOF probability, pooled / I-SPY1 / I-SPY2)
  cv_differences.csv      paired AUROC differences between models, per repeat (descriptive; tests are in 06)
  selected_features_<model>_k{k}.csv   how often LASSO picked each feature over the 25 folds
  <model>_k{k}_<classifier>.joblib     final pipeline refit on the full development set

Run from code\ :  .venv\Scripts\python.exe src\05_train.py [--k 3] [--jobs 4]
"""
import argparse
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
import cv  # noqa: E402
import paths as P  # noqa: E402

N_CLIN = 9            # the first 9 columns of every feature CSV are pid + clinical info
LASSO_CS = np.logspace(-2.5, 0.5, 13)
L2_CS = np.logspace(-3, 1, 9)
CORR_MAX = 0.95
MODE_MAX = 0.95


# ----------------------------------------------------------------------------- pipeline pieces
class NearConstant(BaseEstimator, TransformerMixin):
    """Drop columns where one value covers >= max_mode_frac of the training patients (or SD = 0)."""

    def __init__(self, max_mode_frac=MODE_MAX):
        self.max_mode_frac = max_mode_frac

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        keep = []
        for j in range(X.shape[1]):
            col = np.round(X[:, j], 10)
            _, counts = np.unique(col, return_counts=True)
            keep.append(counts.max() / len(col) < self.max_mode_frac and col.std() > 0)
        self.mask_ = np.array(keep, bool)
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]


class CorrFilter(BaseEstimator, TransformerMixin):
    """Greedy: walk the columns in order, keep a column only if |r| <= max_r with every column kept so far."""

    def __init__(self, max_r=CORR_MAX):
        self.max_r = max_r

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        kept = []
        for j in range(X.shape[1]):
            if not kept or r[j, kept].max() <= self.max_r:
                kept.append(j)
        self.mask_ = np.zeros(X.shape[1], bool)
        self.mask_[kept] = True
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]


class LassoSelect(BaseEstimator, TransformerMixin):
    """L1 logistic regression, C chosen by inner stratified 5-fold CV (AUROC); keep non-zero coefficients."""

    def __init__(self, Cs=LASSO_CS, seed=P.SEED):
        self.Cs = Cs
        self.seed = seed

    def fit(self, X, y):
        m = LogisticRegressionCV(Cs=self.Cs, l1_ratios=(1.0,), solver="liblinear", scoring="roc_auc",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=self.seed),
                                 class_weight="balanced", max_iter=5000, random_state=self.seed)
        m.fit(X, y)
        coef = np.abs(m.coef_.ravel())
        self.C_ = float(m.C_[0])
        self.mask_ = coef > 0
        if not self.mask_.any():          # nothing survived: keep the single strongest feature
            self.mask_[np.argmax(np.abs(np.corrcoef(np.c_[X, y], rowvar=False)[-1, :-1]))] = True
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]


def l2_logistic():
    return LogisticRegressionCV(Cs=L2_CS, l1_ratios=(0.0,), solver="liblinear", scoring="roc_auc",
                                cv=StratifiedKFold(5, shuffle=True, random_state=P.SEED),
                                class_weight="balanced", max_iter=5000, random_state=P.SEED)


def make_pipeline(kind, clf):
    if clf == "lr" and kind == "small":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", l2_logistic())])
    if clf == "lr":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                                       random_state=P.SEED)),
                         ("nearconst", NearConstant()),
                         ("corr", CorrFilter()),
                         ("lasso", LassoSelect()),
                         ("clf", l2_logistic())])
    steps = [("impute", SimpleImputer(strategy="median"))]
    if kind != "small":
        steps += [("nearconst", NearConstant()), ("corr", CorrFilter())]
    steps.append(("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.5, min_child_weight=3, reg_lambda=1.0,
                                       eval_metric="logloss", n_jobs=1, random_state=P.SEED)))
    return Pipeline(steps)


def fit_pipe(pipe, X, y):
    params = {}
    if isinstance(pipe.named_steps["clf"], XGBClassifier):
        params["clf__sample_weight"] = compute_sample_weight("balanced", y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y, **params)
    return pipe


def kept_names(pipe, names):
    """Feature names that reach the classifier."""
    names = np.asarray(names)
    for step in ["nearconst", "corr", "lasso"]:
        if step in pipe.named_steps:
            names = names[pipe.named_steps[step].mask_]
    return list(names)


# ----------------------------------------------------------------------------- data
def arm_group(arm):
    """Plan section 7: HER2-targeted / immunotherapy / platinum / other."""
    if pd.isna(arm):
        return np.nan
    a = arm.lower()
    if any(s in a for s in ["trastuzumab", "pertuzumab", "t-dm1", "neratinib"]):
        return "her2_targeted"
    if "pembrolizumab" in a:
        return "immunotherapy"
    if "carboplatin" in a:
        return "platinum"
    return "other"


def load_data(k):
    """Development patients only, with one feature matrix per model."""
    fdir = P.OUT / "features" / f"k{k}"
    blocks = {b: pd.read_csv(fdir / f"{b}.csv") for b in ["pre", "early", "late", "shape"]}
    dev = cv.dev_patients(cv.load_index())[["pid", "dataset", "split", "pCR", "subtype"]]
    assert not dev["split"].eq("test").any()
    base = blocks["pre"].iloc[:, :N_CLIN].drop(columns=["dataset", "split", "pCR", "subtype"])
    dev = dev.merge(base, on="pid", how="left", validate="1:1")
    for b, df in blocks.items():
        blocks[b] = dev[["pid"]].merge(df.drop(columns=df.columns[1:N_CLIN]), on="pid", how="left",
                                       validate="1:1").drop(columns="pid")
        assert len(blocks[b]) == len(dev) and blocks[b].notna().any(axis=1).all(), b

    clin = pd.DataFrame({"HR": dev["HR"], "HER2": dev["HER2"], "age": dev["age"],
                         "cohort_ispy2": (dev["dataset"] == "spy2").astype(float)})
    X = {"clinical": clin,
         "pre": blocks["pre"], "early": blocks["early"], "late": blocks["late"],
         "fused": pd.concat([blocks["pre"], blocks["early"], blocks["late"], blocks["shape"]], axis=1)}
    g = dev["arm"].map(arm_group)
    arm = pd.get_dummies(g, prefix="arm", dtype=float).drop(columns="arm_other")   # "other" = reference
    X["clinical_arm"] = pd.concat([clin.drop(columns="cohort_ispy2"), arm], axis=1)
    return dev, X


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
    out = P.OUT / "models"
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


if __name__ == "__main__":
    main()
