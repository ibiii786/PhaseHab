r"""
02_preprocess.py - Resample, normalise and crop every patient (see config/analysis_plan.md, sections 3-4).

For each patient:
  1. Load pre / early / late + mask (NIfTI arrays are (z, y, x)).
  2. Set the REAL spacing from the metadata (headers wrongly say 1 mm).
  3. Resample to 1 mm isotropic: images B-spline, mask nearest neighbour on mask > 0.
  4. Normalise: foreground = pre voxels above Otsu (whole crop); m, s = mean/std of pre over
     the foreground; apply (x - m) / s to ALL three phases.
  5. Crop to the tumour bounding box + 10 mm margin.
  6. Save outputs/preprocessed/<pid>.npz and a row in index.csv.
QC overlays (3 phases + mask outline) for 20 random patients + the watch patients.

Run from code\ :
  .venv\Scripts\python.exe src\02_preprocess.py --pilot 50     (50 training patients)
  .venv\Scripts\python.exe src\02_preprocess.py                (all patients)
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import paths as P  # noqa: E402

OUT = P.OUT / "preprocessed"
QC_DIR = P.OUT / "qc" / "preprocess"
NEW_SPACING = 1.0   # mm, isotropic
MARGIN_MM = 10      # crop margin around the tumour
N_QC_RANDOM = 20
WATCH = P.EXCLUDE_ANALYSIS | P.QC_CHECK | set(P.PHASE_OVERRIDE)


# ---------------------------------------------------------------------------
def load_array(path):
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def to_sitk(arr, xy, z):
    """numpy (z, y, x) -> SimpleITK image with the real spacing (x, y, z)."""
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((float(xy), float(xy), float(z)))
    return img


def resample(img, interp):
    old_sp = np.array(img.GetSpacing())
    old_size = np.array(img.GetSize())
    new_size = [int(round(n)) for n in old_size * old_sp / NEW_SPACING]
    return sitk.Resample(img, new_size, sitk.Transform(), interp, img.GetOrigin(),
                         (NEW_SPACING,) * 3, img.GetDirection(), 0.0, img.GetPixelID())


def qc_png(pid, row, pre, early, late, mask, m, s, note=""):
    """Middle tumour slice: 3 phases side by side with the mask outline."""
    zs = np.where(mask.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return
    z = int(zs[len(zs) // 2])
    stack = np.stack([pre[z], early[z], late[z]])
    vmin, vmax = np.percentile(stack, [1, 99.5])  # SAME window for all 3 phases
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for a, im, name in zip(ax, stack, ["pre", "early", "late"]):
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        a.contour(mask[z], levels=[0.5], colors="red", linewidths=0.8)
        a.set_title(name)
        a.axis("off")
    fig.suptitle(f"{pid} ({row['dataset']}, {row['split']}, pCR={row['pCR']})  slice {z}/{mask.shape[0]}  "
                 f"tumour {int(mask.sum())} mm3  m={m:.0f} s={s:.0f} {note}", fontsize=9)
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{pid}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def process(row, make_qc):
    t0 = time.time()
    pid, ds = row["pid"], row["dataset"]
    xy, zt = row["xy_spacing"], row["slice_thick"]

    # 1-3: load and resample
    imgs = {}
    acqs = {phase: P.phase_acq(row, phase) for phase in P.PHASES}  # includes QC corrections
    for phase, acq in acqs.items():
        arr = load_array(P.scan_path(pid, ds, acq))
        imgs[phase] = sitk.GetArrayFromImage(resample(to_sitk(arr, xy, zt), sitk.sitkBSpline))
    mraw = (load_array(P.mask_path(pid, ds)) > 0).astype(np.uint8)
    mask = sitk.GetArrayFromImage(resample(to_sitk(mraw, xy, zt), sitk.sitkNearestNeighbor))

    # 4: normalisation from the pre-contrast image only
    pre_img = sitk.GetImageFromArray(imgs["pre"])
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.Execute(pre_img)
    thr = otsu.GetThreshold()
    fg = imgs["pre"] > thr
    m = float(imgs["pre"][fg].mean())
    s = float(imgs["pre"][fg].std())
    for k in imgs:
        imgs[k] = ((imgs[k] - m) / s).astype(np.float32)

    if make_qc:
        note = "[WATCH]" if pid in WATCH else ""
        qc_png(pid, row, imgs["pre"], imgs["early"], imgs["late"], mask, m, s, note)

    # 5: crop around the tumour + margin
    nz = np.argwhere(mask > 0)
    full_shape = mask.shape
    if len(nz):
        lo = np.maximum(nz.min(0) - MARGIN_MM, 0)
        hi = np.minimum(nz.max(0) + MARGIN_MM + 1, full_shape)
    else:
        lo, hi = np.zeros(3, int), np.array(full_shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))

    # 6: save
    np.savez_compressed(
        OUT / f"{pid}.npz",
        pre=imgs["pre"][sl], early=imgs["early"][sl], late=imgs["late"][sl],
        mask=mask[sl].astype(np.uint8),
        spacing=np.array([NEW_SPACING] * 3), m=m, s=s, otsu=thr,
        orig_spacing=np.array([zt, xy, xy]), crop_lo=lo, crop_hi=hi, full_shape=np.array(full_shape))

    tum = int(mask.sum())
    return {
        "pid": pid, "dataset": ds, "split": row["split"], "pCR": row["pCR"],
        "subtype": row["HR_HER2_STATUS"],
        "use": pid not in P.EXCLUDE_ANALYSIS,
        "acq": "/".join(str(acqs[p]) for p in P.PHASES),
        "shape": "x".join(map(str, mask[sl].shape)), "full_shape": "x".join(map(str, full_shape)),
        "tumour_voxels": tum, "tumour_vox_orig": int(mraw.sum()),
        "vol_ratio": round(tum / (mraw.sum() * xy * xy * zt), 4) if mraw.sum() else np.nan,
        "m": round(m, 3), "s": round(s, 3), "otsu": round(thr, 3),
        "fg_fraction": round(float(fg.mean()), 4),
        "early_mean_in_tumour": round(float(imgs["early"][mask > 0].mean()), 3) if tum else np.nan,
        "pre_mean_in_tumour": round(float(imgs["pre"][mask > 0].mean()), 3) if tum else np.nan,
        "late_mean_in_tumour": round(float(imgs["late"][mask > 0].mean()), 3) if tum else np.nan,
        "seconds": round(time.time() - t0, 2),
    }


def safe_process(row, make_qc):
    try:
        return process(row, make_qc)
    except Exception as e:  # noqa: BLE001
        return {"pid": row["pid"], "dataset": row["dataset"], "split": row["split"], "error": repr(e)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="only N random training patients")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="+", default=None,
                    help="re-run only these pids and update their rows in index.csv")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(P.META_MAIN)
    meta = meta[meta["dataset"].isin(["spy1", "spy2"]) & ~meta["pid"].isin(P.EXCLUDE)].copy()
    meta["split"] = meta["test"].map(P.SPLIT_NAMES)
    meta["pCR"] = meta["pCR"].astype(int)

    if args.only:
        meta = meta[meta["pid"].isin(args.only)]
        qc_ids = set(meta["pid"])
        index_name = "index.csv"
    elif args.pilot:
        meta = meta[meta["split"] == "train"].sample(args.pilot, random_state=P.SEED)
        qc_ids = set()
        index_name = "index_pilot.csv"
    else:
        rng = np.random.default_rng(P.SEED)
        normal = meta[~meta["pid"].isin(WATCH)]["pid"].to_numpy()
        qc_ids = set(rng.choice(normal, N_QC_RANDOM, replace=False)) | WATCH
        index_name = "index.csv"

    rows = meta.to_dict("records")
    print(f"Preprocessing {len(rows)} patients with {args.jobs} workers ...")
    t0 = time.time()
    gen = Parallel(n_jobs=args.jobs, return_as="generator")(
        delayed(safe_process)(r, r["pid"] in qc_ids) for r in rows)
    res = pd.DataFrame(list(tqdm(gen, total=len(rows), unit="pt")))
    wall = time.time() - t0

    if args.only and (OUT / index_name).exists():
        # replace just these rows in the existing index, keep the original order
        old = pd.read_csv(OUT / index_name)
        full = pd.concat([old[~old["pid"].isin(res["pid"])], res])
        full = full.set_index("pid").loc[old["pid"]].reset_index()
        full.to_csv(OUT / index_name, index=False)
    else:
        res.to_csv(OUT / index_name, index=False)
    n_err = int(res["error"].notna().sum()) if "error" in res else 0
    ok = res[res.get("error", pd.Series(index=res.index, dtype=object)).isna()]
    print(f"Done: {len(ok)} ok, {n_err} errors. Wall time {wall/60:.1f} min, "
          f"{wall/len(rows):.2f} s/patient (parallel), {ok['seconds'].mean():.2f} s/patient (one worker).")
    print(f"Volume ratio after resampling (should be ~1): median {ok['vol_ratio'].median():.3f}, "
          f"min {ok['vol_ratio'].min():.3f}, max {ok['vol_ratio'].max():.3f}")
    size_mb = sum(f.stat().st_size for f in OUT.glob("*.npz")) / 1e6
    print(f"Disk used by .npz files: {size_mb:.0f} MB")
    if n_err:
        print(res[res["error"].notna()][["pid", "error"]].to_string())


if __name__ == "__main__":
    main()
