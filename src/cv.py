"""Shared cohort + cross-validation helpers, so every script uses the SAME patients and folds."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import paths as P

N_SPLITS, N_REPEATS = 5, 5


def load_index():
    """Preprocessed patients that are used in the analysis (index.csv, use == True)."""
    idx = pd.read_csv(P.OUT / "preprocessed" / "index.csv")
    return idx[idx["use"]].reset_index(drop=True)


def dev_patients(idx):
    """Development set = train + val. The test set is never returned here."""
    return idx[idx["split"].isin(["train", "val"])].reset_index(drop=True)


def strata(df):
    """pCR x subtype. Missing subtype -> HRposHER2neg (stratification only, see analysis plan)."""
    sub = df["subtype"].fillna("HRposHER2neg")
    return df["pCR"].astype(int).astype(str) + "_" + sub


def dev_folds(dev):
    """List of (repeat, fold, train_idx, val_idx) for repeated stratified 5-fold CV (seed 42)."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=P.SEED)
    out = []
    for i, (tr, va) in enumerate(rskf.split(np.zeros(len(dev)), strata(dev))):
        out.append((i // N_SPLITS, i % N_SPLITS, tr, va))
    return out
