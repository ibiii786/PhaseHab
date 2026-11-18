r"""
04_features.py - PyRadiomics features per phase x habitat (config/analysis_plan.md, section 6).

For every patient:
  - for each phase (pre, early, late) and each habitat h = 1..k:
      run PyRadiomics (config/radiomics.yaml) on the habitat region.
      Habitat < 100 voxels -> features left empty (NaN).
      Names: <phase>_h<h>_<class>_<feature>, e.g. early_h2_glcm_Contrast
  - shape features once on the whole tumour mask: shape_<feature>
    + habitat volume fractions: shape_habfrac_h<h>
Each patient is saved to outputs/features/k{k}/parts/<pid>.json as soon as it is done, so a crash
loses nothing and a restart skips finished patients (only if the yaml did not change).
Every 100 patients and at the end the parts are combined into pre.csv, early.csv, late.csv, shape.csv.

Run from code\ :
  .venv\Scripts\python.exe src\04_features.py --k 3 --check-bins     (choose binWidth, no extraction)
  .venv\Scripts\python.exe src\04_features.py --k 3 --pilot 20
  .venv\Scripts\python.exe src\04_features.py --k 3 --jobs 4         (all; used by run_features.bat)
  .venv\Scripts\python.exe src\04_features.py --k 3 --combine-only
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import cv  # noqa: E402
import paths as P  # noqa: E402

PRE = P.OUT / "preprocessed"
YAML = P.CODE / "config" / "radiomics.yaml"
MIN_VOX = 100
CLIN_COLS = ["pid", "dataset", "split", "pCR", "subtype", "HR", "HER2", "age", "arm"]

_EXT = {}  # per-worker cache of PyRadiomics extractors


def yaml_hash():
    return hashlib.md5(YAML.read_bytes()).hexdigest()[:10]


def extractors():
    """Create the two extractors once per worker process."""
    if not _EXT:
        import SimpleITK as sitk
        from radiomics import featureextractor
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        # glcm_MCC uses eigenvalues; PyRadiomics drops a tiny imaginary part (known, harmless)
        import warnings
        warnings.filterwarnings("ignore", message="Casting complex values")
        logging.getLogger("radiomics").setLevel(logging.ERROR)
        tex = featureextractor.RadiomicsFeatureExtractor(str(YAML))
        shp = featureextractor.RadiomicsFeatureExtractor(str(YAML))
        shp.disableAllFeatures()
        shp.enableFeatureClassByName("shape")
        _EXT.update(tex=tex, shp=shp, sitk=sitk)
    return _EXT


def to_img(arr, sitk):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def clean(result, prefix):
    """PyRadiomics dict -> {prefix_class_feature: float}, dropping diagnostics."""
    out = {}
    for key, val in result.items():
        if key.startswith("diagnostics_"):
            continue
        name = key.replace("original_", "", 1)      # e.g. glcm_Contrast, shape_Sphericity
        out[f"{prefix}_{name}" if prefix else name] = float(val)
    return out


def process(pid, k, out_dir, h):
    t0 = time.time()
    E = extractors()
    sitk = E["sitk"]
    z = np.load(PRE / f"{pid}.npz")
    hab = np.load(P.OUT / "habitats" / f"k{k}" / f"{pid}.npz")["hab"]
    res = {"pid": pid, "yaml": h, "pre": {}, "early": {}, "late": {}, "shape": {}, "errors": []}

    masks = {}
    for hh in range(1, k + 1):
        n = int((hab == hh).sum())
        res["shape"][f"shape_habfrac_h{hh}"] = n / max(int((hab > 0).sum()), 1)
        if n >= MIN_VOX:
            masks[hh] = to_img((hab == hh).astype(np.uint8), sitk)

    for phase in P.PHASES:
        img = to_img(z[phase].astype(np.float32), sitk)
        for hh, m in masks.items():
            try:
                res[phase].update(clean(E["tex"].execute(img, m, label=1), f"{phase}_h{hh}"))
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"{phase}_h{hh}: {e!r}")

    try:
        whole = to_img((hab > 0).astype(np.uint8), sitk)
        # shape does not depend on intensities; the image is only needed for geometry
        shp = E["shp"].execute(to_img(z["early"].astype(np.float32), sitk), whole, label=1)
        res["shape"].update(clean(shp, None))     # names: shape_<feature>
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"shape: {e!r}")

    res["seconds"] = round(time.time() - t0, 2)
    (out_dir / "parts" / f"{pid}.json").write_text(json.dumps(res))
    return pid, res["seconds"], len(res["errors"])


def clinical_table():
    idx = cv.load_index()
    meta = pd.read_csv(P.META_MAIN)[["pid", "HR", "HER2", "age"]]
    tcia = pd.read_csv(P.META_SPY2)[["pid", "Arm"]].rename(columns={"Arm": "arm"})
    return idx[["pid", "dataset", "split", "pCR", "subtype"]].merge(meta, on="pid", how="left") \
        .merge(tcia, on="pid", how="left")[CLIN_COLS]


def combine(out_dir, h):
    """parts/*.json -> pre.csv, early.csv, late.csv, shape.csv (only parts made with the current yaml)."""
    clin = clinical_table()
    blocks = {b: [] for b in ["pre", "early", "late", "shape"]}
    n = 0
    for f in sorted((out_dir / "parts").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("yaml") != h:
            continue
        n += 1
        for b in blocks:
            blocks[b].append({"pid": r["pid"], **r[b]})
    for b, rows in blocks.items():
        if not rows:
            continue
        df = clin.merge(pd.DataFrame(rows), on="pid", how="inner")
        df.to_csv(out_dir / f"{b}.csv", index=False)
    return n


def check_bins(k):
    """Range of normalised intensity inside each habitat region, on TRAINING patients only."""
    idx = cv.load_index()
    tr = idx[idx["split"] == "train"]
    rows = []
    for pid in tqdm(tr["pid"], desc="Checking intensity ranges"):
        z = np.load(PRE / f"{pid}.npz")
        hab = np.load(P.OUT / "habitats" / f"k{k}" / f"{pid}.npz")["hab"]
        for hh in range(1, k + 1):
            m = hab == hh
            if m.sum() < MIN_VOX:
                continue
            for ph in P.PHASES:
                v = z[ph][m]
                rows.append({"pid": pid, "phase": ph, "habitat": hh, "range": float(v.max() - v.min())})
    d = pd.DataFrame(rows)
    out = []
    for bw in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        bins = np.ceil(d["range"] / bw)
        for ph in P.PHASES + ["all"]:
            b = bins if ph == "all" else bins[d["phase"] == ph]
            out.append({"binWidth": bw, "phase": ph, "median_bins": float(b.median()),
                        "p5_bins": float(b.quantile(0.05)), "p95_bins": float(b.quantile(0.95)),
                        "%_in_30_130": round(100 * ((b >= 30) & (b <= 130)).mean(), 1)})
    res = pd.DataFrame(out)
    dest = P.OUT / "features" / f"k{k}"
    dest.mkdir(parents=True, exist_ok=True)
    res.to_csv(dest / "bin_check.csv", index=False)
    print(d.groupby("phase")["range"].describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_string())
    print(res.to_string(index=False))


def setup_log(out_dir):
    log = logging.getLogger("phasehab")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(out_dir / "run_features.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--check-bins", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    args = ap.parse_args()

    if args.check_bins:
        check_bins(args.k)
        return

    out_dir = P.OUT / "features" / f"k{args.k}"
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    h = yaml_hash()
    log = setup_log(out_dir)

    if args.combine_only:
        print(f"Combined {combine(out_dir, h)} patients")
        return

    idx = cv.load_index()
    if args.pilot:
        todo_all = idx[idx["split"] == "train"].sample(args.pilot, random_state=P.SEED)["pid"].tolist()
    else:
        todo_all = idx["pid"].tolist()

    def done(pid):
        f = out_dir / "parts" / f"{pid}.json"
        if not f.exists():
            return False
        try:
            return json.loads(f.read_text()).get("yaml") == h
        except Exception:  # noqa: BLE001  (half-written file after a crash -> redo)
            return False

    todo = [p for p in todo_all if not done(p)]
    msg = (f"START k={args.k} jobs={args.jobs} yaml={h}: {len(todo_all)} patients, "
           f"{len(todo_all) - len(todo)} already done, {len(todo)} to do")
    print(msg)
    log.info(msg)
    if not todo:
        n = combine(out_dir, h)
        print(f"Nothing to do. Combined {n} patients into {out_dir}")
        return

    t0 = time.time()
    gen = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(process)(p, args.k, out_dir, h) for p in todo)
    secs = []
    for i, (pid, s, n_err) in enumerate(tqdm(gen, total=len(todo), unit="pt"), 1):
        secs.append(s)
        log.info(f"{pid} done in {s:.1f}s ({i}/{len(todo)})" + (f"  ERRORS: {n_err}" if n_err else ""))
        if i % 100 == 0:
            n = combine(out_dir, h)
            log.info(f"checkpoint: combined {n} patients into CSVs")
    wall = time.time() - t0
    n = combine(out_dir, h)
    msg = (f"END: {len(todo)} patients in {wall/60:.1f} min; {np.mean(secs):.1f} s/patient per worker, "
           f"{wall/len(todo):.1f} s/patient overall. CSVs now hold {n} patients.")
    print(msg)
    log.info(msg)


if __name__ == "__main__":
    main()
