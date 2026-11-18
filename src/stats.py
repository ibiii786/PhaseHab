"""Shared statistics: DeLong test for paired AUROCs, Holm correction, stratified bootstrap, calibration.

DeLong: fast algorithm of Sun & Xu (2014), IEEE Signal Processing Letters 21(11):1389-1393.
Used by 06_compare.py (CV) and 08_final_test.py (test set).
"""
import numpy as np
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss


def _midrank(x):
    """Mid-ranks (ties share the average rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y, preds):
    """preds: (n_models, n_patients). Returns AUROCs and their covariance matrix."""
    y = np.asarray(y).astype(int)
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ci(y, p, alpha=0.05):
    """AUROC with DeLong 95% CI."""
    auc, cov = _delong_components(y, np.atleast_2d(p))
    se = np.sqrt(cov[0, 0])
    z = st.norm.ppf(1 - alpha / 2)
    return float(auc[0]), float(max(0, auc[0] - z * se)), float(min(1, auc[0] + z * se))


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs (same patients). Returns dict."""
    aucs, cov = _delong_components(y, np.vstack([p1, p2]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 1e-12))
    z = diff / se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "diff_lo": float(diff - 1.96 * se), "diff_hi": float(diff + 1.96 * se),
            "z": float(z), "p": float(2 * st.norm.sf(abs(z)))}


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fast_auc(y, p):
    """AUROC by ranks (no sklearn overhead), for the bootstrap."""
    r = _midrank(np.asarray(p, float))
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_indices(y, n_boot=2000, seed=42):
    """Stratified bootstrap samples (keeps the number of pCR+ and pCR- patients fixed)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(n_boot)]


def boot_ci(y, p, idx_list):
    """Percentile 95% CIs for AUROC, AUPRC and Brier."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    vals = np.array([(fast_auc(y[i], p[i]), average_precision_score(y[i], p[i]), brier_score_loss(y[i], p[i]))
                     for i in idx_list])
    lo, hi = np.percentile(vals, [2.5, 97.5], axis=0)
    return {"AUROC_lo": lo[0], "AUROC_hi": hi[0], "AUPRC_lo": lo[1], "AUPRC_hi": hi[1],
            "Brier_lo": lo[2], "Brier_hi": hi[2]}


def calibration(y, p):
    """Calibration-in-the-large (mean predicted - observed rate) and calibration slope (logistic on logit p)."""
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y).coef_[0, 0]
    return {"mean_pred": float(p.mean()), "obs_rate": float(y.mean()),
            "citl": float(p.mean() - y.mean()), "slope": float(slope)}


def youden_threshold(y, p):
    """Threshold that maximises sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(min(thr[j], 1.0))
