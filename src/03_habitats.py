import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import cv
import paths as P

PRE = P.OUT / "preprocessed"
OUT = P.OUT / "habitats"
QC = P.OUT / "qc" / "habitats"
KS = [2, 3, 4]
N_SAMPLE = 2000
MIN_VOX = 100
JOBS = 4
HAB_COLOURS = ["#2c7bb6", "#fdae61", "#d7191c", "#1a9641"]


def tumour_voxels(pid):
    z = np.load(PRE / f"{pid}.npz")
    m = z["mask"] > 0
    return np.stack([z["pre"][m], z["early"][m], z["late"][m]], axis=1).astype(np.float32)


def sample_patient(pid, seed):
    x = tumour_voxels(pid)
    rng = np.random.default_rng(seed)
    if len(x) > N_SAMPLE:
        x = x[rng.choice(len(x), N_SAMPLE, replace=False)]
    return pid, x


def fit_kmeans(x, k):
    km = KMeans(n_clusters=k, random_state=P.SEED, n_init=10)
    km.fit(x)
    order = np.argsort(km.cluster_centers_[:, 1])
    km.cluster_centers_ = km.cluster_centers_[order]
    return km


def assign(centres, x):
    d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return d.argmin(1) + 1


def describe(c, all_centres):
    pre, early, late = c
    lo, hi = np.percentile(all_centres[:, 1], [33, 67]) if len(all_centres) > 2 else (early, early)
    level = "low" if early <= lo else ("high" if early >= hi else "medium")
    if len(all_centres) == 2:
        level = "low" if early == all_centres[:, 1].min() else "high"
    change = late - early
    if change < -0.3:
        curve = "wash-out (late darker than early)"
    elif change > 0.3:
        curve = "persistent (still rising on late)"
    else:
        curve = "plateau (late ~ early)"
    return f"{level} enhancement, {curve}"


def map_patient(pid, centres_by_k):
    z = np.load(PRE / f"{pid}.npz")
    m = z["mask"] > 0
    x = np.stack([z["pre"][m], z["early"][m], z["late"][m]], axis=1)
    rows = {}
    for k, cen in centres_by_k.items():
        hab = np.zeros(m.shape, np.uint8)
        hab[m] = assign(cen, x)
        np.savez_compressed(OUT / f"k{k}" / f"{pid}.npz", hab=hab)
        counts = np.bincount(hab[m], minlength=k + 1)[1:]
        rows[k] = counts
    return pid, rows


def qc_png(pid, k, row):
    z = np.load(PRE / f"{pid}.npz")
    hab = np.load(OUT / f"k{k}" / f"{pid}.npz")["hab"]
    zz = int(np.argmax((hab > 0).sum(axis=(1, 2))))
    stack = np.stack([z["pre"][zz], z["early"][zz], z["late"][zz]])
    vmin, vmax = np.percentile(stack, [1, 99.5])
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    for a, im, name in zip(ax[:3], stack, ["pre", "early", "late"]):
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        a.contour(hab[zz] > 0, levels=[0.5], colors="yellow", linewidths=0.7)
        a.set_title(name)
        a.axis("off")
    ax[3].imshow(stack[1], cmap="gray", vmin=vmin, vmax=vmax)
    ov = np.ma.masked_where(hab[zz] == 0, hab[zz])
    ax[3].imshow(ov, cmap=ListedColormap(HAB_COLOURS[:k]), vmin=0.5, vmax=k + 0.5, alpha=0.65,
                 interpolation="nearest")
    ax[3].set_title(f"habitats k={k} (" + ", ".join(f"h{i+1}" for i in range(k)) + ")")
    ax[3].axis("off")
    pct = ", ".join(f"h{i+1} {row[f'pct_h{i+1}']:.0f}%" for i in range(k))
    fig.suptitle(f"{pid} ({row['split']}, pCR={row['pCR']})  slice {zz}  |  {pct}   "
                 f"[blue = h1 low enhancement ... red/green = highest]", fontsize=9)
    fig.tight_layout()
    fig.savefig(QC / f"{pid}_k{k}.png", dpi=100)
    plt.close(fig)


def main():
    t0 = time.time()
    for k in KS:
        (OUT / f"k{k}").mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    idx = cv.load_index()
    dev = cv.dev_patients(idx)
    print(f"Patients: {len(idx)} in analysis, {len(dev)} development (train + val) used to fit K-means")

    res = Parallel(n_jobs=JOBS)(delayed(sample_patient)(p, P.SEED + i)
                                for i, p in enumerate(tqdm(dev["pid"], desc="Sampling voxels")))
    samples = dict(res)
    X = np.concatenate([samples[p] for p in dev["pid"]])
    owner = np.concatenate([np.full(len(samples[p]), i) for i, p in enumerate(dev["pid"])])
    print(f"Fitting on {len(X):,} voxels")

    rng = np.random.default_rng(P.SEED)
    sil_idx = rng.choice(len(X), min(20000, len(X)), replace=False)
    models, report = {}, []
    for k in KS:
        km = fit_kmeans(X, k)
        joblib.dump(km, OUT / f"kmeans_k{k}.joblib")
        models[k] = km
        lab = assign(km.cluster_centers_, X)
        sil = silhouette_score(X[sil_idx], lab[sil_idx])
        cen = km.cluster_centers_
        for h in range(k):
            report.append({"k": k, "habitat": h + 1, "pre": cen[h, 0], "early": cen[h, 1], "late": cen[h, 2],
                           "wash_in": cen[h, 1] - cen[h, 0], "late_minus_early": cen[h, 2] - cen[h, 1],
                           "share_of_sampled_voxels_%": 100 * (lab == h + 1).mean(),
                           "meaning": describe(cen[h], cen), "silhouette": sil})
    centres = pd.DataFrame(report).round(3)
    centres.to_csv(OUT / "centres.csv", index=False)
    print(centres.to_string(index=False))

    folds = [f for f in cv.dev_folds(dev) if f[0] == 0]
    stab = []
    for k in KS:
        main_lab = assign(models[k].cluster_centers_, X)
        for _, fold, tr, _va in folds:
            sel = np.isin(owner, tr)
            km_f = fit_kmeans(X[sel], k)
            stab.append({"k": k, "fold": fold,
                         "ARI": adjusted_rand_score(main_lab, assign(km_f.cluster_centers_, X)),
                         "max_centre_shift": float(np.abs(km_f.cluster_centers_ - models[k].cluster_centers_).max())})
    stab = pd.DataFrame(stab).round(4)
    stab.to_csv(OUT / "stability_ari.csv", index=False)
    print(stab.groupby("k")[["ARI", "max_centre_shift"]].agg(["min", "mean"]).round(3).to_string())

    cen_by_k = {k: models[k].cluster_centers_.astype(np.float32) for k in KS}
    res = Parallel(n_jobs=JOBS)(delayed(map_patient)(p, cen_by_k)
                                for p in tqdm(idx["pid"], desc="Assigning habitats"))
    counts = dict(res)
    small_summary = []
    for k in KS:
        rows = []
        for _, r in idx.iterrows():
            c = counts[r["pid"]][k]
            row = {"pid": r["pid"], "dataset": r["dataset"], "split": r["split"], "pCR": r["pCR"],
                   "tumour_voxels": int(c.sum())}
            for h in range(k):
                row[f"vox_h{h+1}"] = int(c[h])
            for h in range(k):
                row[f"pct_h{h+1}"] = round(100 * c[h] / c.sum(), 2)
            for h in range(k):
                row[f"small_h{h+1}"] = bool(c[h] < MIN_VOX)
            row["any_small"] = bool((c < MIN_VOX).any())
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(OUT / f"habitat_table_k{k}.csv", index=False)
        s = {"k": k, "patients_with_any_habitat_<100": int(tab["any_small"].sum()),
             "%_patients": round(100 * tab["any_small"].mean(), 1)}
        for h in range(k):
            s[f"h{h+1}_<100"] = int(tab[f"small_h{h+1}"].sum())
            s[f"h{h+1}_median_%"] = round(tab[f"pct_h{h+1}"].median(), 1)
        small_summary.append(s)
    small = pd.DataFrame(small_summary)
    small.to_csv(OUT / "small_habitats_summary.csv", index=False)
    print(small.to_string(index=False))

    rng = np.random.default_rng(P.SEED)
    qc_ids = list(rng.choice(idx["pid"], 10, replace=False))
    for k in KS:
        tab = pd.read_csv(OUT / f"habitat_table_k{k}.csv").set_index("pid")
        for p in (qc_ids if k == 3 else qc_ids[:3]):
            qc_png(p, k, tab.loc[p].to_dict() | {"pid": p})
    print(f"Done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
