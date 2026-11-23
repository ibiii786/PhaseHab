import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import paths as P

N_SPLITS, N_REPEATS = 5, 5


def load_index():
    idx = pd.read_csv(P.OUT / "preprocessed" / "index.csv")
    return idx[idx["use"]].reset_index(drop=True)


def dev_patients(idx):
    return idx[idx["split"].isin(["train", "val"])].reset_index(drop=True)


def strata(df):
    sub = df["subtype"].fillna("HRposHER2neg")
    return df["pCR"].astype(int).astype(str) + "_" + sub


def dev_folds(dev):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=P.SEED)
    out = []
    for i, (tr, va) in enumerate(rskf.split(np.zeros(len(dev)), strata(dev))):
        out.append((i // N_SPLITS, i % N_SPLITS, tr, va))
    return out
