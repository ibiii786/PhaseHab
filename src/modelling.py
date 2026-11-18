"""Shared model code: the sklearn pipeline pieces and the data loading used by 05_train.py, 07_shap.py,
08_final_test.py and the web app. The saved .joblib models need these classes, so keep them here
(moving or renaming them breaks loading of saved models)."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
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
