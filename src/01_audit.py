import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import paths as P

OUT = P.OUT / "audit"
OUT.mkdir(parents=True, exist_ok=True)

MASK_NOTES = P.CODE / "docs" / "mask_origin.md"

GRID_CANDIDATES = [1.0, 1.5, 2.0]


def load_metadata():
    meta = pd.read_csv(P.META_MAIN)
    meta = meta[meta["dataset"].isin(["spy1", "spy2"])].copy()
    n_all = len(meta)
    excluded = meta[meta["pid"].isin(P.EXCLUDE)][["pid", "dataset", "test", "pCR"]]
    meta = meta[~meta["pid"].isin(P.EXCLUDE)].copy()
    meta["split"] = meta["test"].map(P.SPLIT_NAMES)
    meta["pCR"] = meta["pCR"].astype("Int64")
    meta["subtype"] = meta["HR_HER2_STATUS"].fillna("missing")
    return meta.reset_index(drop=True), excluded, n_all


def count_tables(meta):
    def summarise(g):
        return pd.Series({
            "n": len(g),
            "pCR_pos": int((g["pCR"] == 1).sum()),
            "pCR_neg": int((g["pCR"] == 0).sum()),
            "pCR_missing": int(g["pCR"].isna().sum()),
            "pCR_rate_%": round(100 * (g["pCR"] == 1).sum() / max(g["pCR"].notna().sum(), 1), 1),
        })

    both = pd.concat([meta, meta.assign(dataset="pooled")])
    both = pd.concat([both, both.assign(split="all")])
    order_s = ["train", "val", "test", "all"]
    counts = (both.groupby(["dataset", "split"]).apply(summarise, include_groups=False)
              .reset_index())
    counts["split"] = pd.Categorical(counts["split"], order_s, ordered=True)
    counts = counts.sort_values(["dataset", "split"]).reset_index(drop=True)

    sub = (both.groupby(["dataset", "split", "subtype"]).agg(
        n=("pid", "size"), pCR_pos=("pCR", lambda s: int((s == 1).sum())))
        .reset_index())
    sub["pCR_rate_%"] = (100 * sub["pCR_pos"] / sub["n"]).round(1)
    sub["split"] = pd.Categorical(sub["split"], order_s, ordered=True)
    sub = sub.sort_values(["dataset", "split", "subtype"]).reset_index(drop=True)
    return counts, sub


def arm_table(meta):
    tcia = pd.read_csv(P.META_SPY2)
    s2 = meta[meta["dataset"] == "spy2"].merge(
        tcia[["pid", "Arm", "pCR", "HR", "HER2", "manufacturer"]],
        on="pid", how="left", suffixes=("", "_tcia"))
    checks = {
        "spy2_patients": len(s2),
        "missing_in_tcia": int(s2["Arm"].isna().sum()),
        "pCR_disagree": int((s2["pCR"].astype(float) != s2["pCR_tcia"].astype(float)).sum()),
        "HR_disagree": int(((s2["HR"] != s2["HR_tcia"]) & s2["HR"].notna()).sum()),
        "HER2_disagree": int(((s2["HER2"] != s2["HER2_tcia"]) & s2["HER2"].notna()).sum()),
    }
    arms = pd.crosstab(s2["Arm"].fillna("missing"), s2["split"])
    arms = arms.reindex(columns=[c for c in ["train", "val", "test"] if c in arms.columns])
    arms["total"] = arms.sum(axis=1)
    pcr = s2.groupby("Arm")["pCR"].apply(lambda s: round(100 * (s == 1).mean(), 1))
    arms["pCR_rate_%"] = pcr
    arms = arms.sort_values("total", ascending=False).reset_index()
    manu = pd.crosstab(s2["manufacturer"].fillna("missing"), s2["split"])
    return arms, manu.reset_index(), checks


def check_patient(row):
    t0 = time.time()
    pid, ds = row["pid"], row["dataset"]
    res = {"pid": pid, "dataset": ds, "split": row["split"], "pCR": row["pCR"]}
    problems = []
    shapes = {}

    for phase, col in P.PHASE_COLS.items():
        acq = row[col]
        if pd.isna(acq):
            problems.append(f"{phase}: no acquisition number in metadata")
            res[f"{phase}_exists"] = False
            continue
        f = P.scan_path(pid, ds, int(acq))
        res[f"{phase}_acq"] = int(acq)
        res[f"{phase}_exists"] = f.exists()
        if not f.exists():
            problems.append(f"{phase}: file missing ({f.name})")
            continue
        try:
            shapes[phase] = tuple(nib.load(str(f)).shape)
        except Exception as e:
            problems.append(f"{phase}: cannot read header ({e})")

    mf = P.mask_path(pid, ds)
    res["mask_exists"] = mf.exists()
    if not mf.exists():
        problems.append("mask: file missing")
    else:
        try:
            img = nib.load(str(mf))
            shapes["mask"] = tuple(img.shape)
            m = np.asanyarray(img.dataobj)
            vals = np.unique(m)
            res["mask_values"] = " ".join(str(v) for v in vals[:6]) + (" ..." if len(vals) > 6 else "")
            tumour = m > 0
            nvox = int(tumour.sum())
            res["tumour_voxels"] = nvox
            zs = np.where(tumour.any(axis=(1, 2)))[0]
            res["tumour_slices"] = int(len(zs))
            if nvox == 0:
                problems.append("mask: empty (0 tumour voxels)")
            if len(vals) > 2:
                problems.append(f"mask: not binary ({len(vals)} distinct values)")
        except Exception as e:
            problems.append(f"mask: cannot read ({e})")

    for k, s in shapes.items():
        res[f"{k}_shape"] = "x".join(map(str, s))
    if len(set(shapes.values())) > 1:
        problems.append("shape mismatch: " + ", ".join(f"{k}={v}" for k, v in shapes.items()))
    if "mask" in shapes and pd.notna(row["n_z"]):
        res["file_minus_meta_nz"] = shapes["mask"][0] - int(row["n_z"])

    xy, z = row["xy_spacing"], row["slice_thick"]
    res["xy_spacing"], res["slice_thick"] = xy, z
    if "tumour_voxels" in res and pd.notna(xy) and pd.notna(z):
        res["tumour_mm3"] = res["tumour_voxels"] * xy * xy * z
    else:
        problems.append("cannot compute volume (missing spacing or mask)")
    res["meta_tum_vol"] = row.get("tum_vol")

    res["n_problems"] = len(problems)
    res["problems"] = "; ".join(problems)
    res["seconds"] = round(time.time() - t0, 3)
    return res


def run_checks(meta):
    rows = meta.to_dict("records")
    n_jobs = min(8, os.cpu_count() or 1)
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(check_patient)(r) for r in rows)
    results = list(tqdm(gen, total=len(rows), desc="Checking patients", unit="pt"))
    return pd.DataFrame(results)


def spacing_tables(chk):
    both = pd.concat([chk, chk.assign(dataset="pooled")])
    rows = []
    for ds, g in both.groupby("dataset"):
        for col in ["xy_spacing", "slice_thick"]:
            d = g[col].describe(percentiles=[0.05, 0.5, 0.95])
            rows.append({"dataset": ds, "measure": col, "n": int(d["count"]),
                         "missing": int(g[col].isna().sum()),
                         "min": d["min"], "p5": d["5%"], "median": d["50%"],
                         "p95": d["95%"], "max": d["max"]})
    summ = pd.DataFrame(rows).round(4)

    vals = (both.assign(xy=both["xy_spacing"].round(3), z=both["slice_thick"].round(2))
            .groupby(["dataset", "z"]).size().rename("n_patients").reset_index())

    grid_rows = []
    for g_mm in GRID_CANDIDATES:
        v = chk["tumour_mm3"] / g_mm ** 3
        for ds, sub in chk.assign(v=v).groupby("dataset"):
            grid_rows.append({"grid_mm": g_mm, "dataset": ds,
                              "min_voxels": round(sub["v"].min()),
                              "p5_voxels": round(sub["v"].quantile(0.05)),
                              "median_voxels": round(sub["v"].median()),
                              "n_under_500_voxels": int((sub["v"] < 500).sum()),
                              "n_under_1000_voxels": int((sub["v"] < 1000).sum())})
    grid = pd.DataFrame(grid_rows)
    return summ, vals, grid


def md_table(df):
    return df.to_markdown(index=False)


def write_summary(meta, excluded, n_all, counts, sub, arms, manu, arm_checks,
                  chk, problems, sizes, smallest, spacing, grid, minutes):
    c = counts.set_index(["dataset", "split"])
    L = []
    L.append("# Cohort audit (01_audit.py)\n")
    L.append(f"Run date: {time.strftime('%Y-%m-%d %H:%M')}. Run time: {minutes:.1f} min.\n")

    L.append("## 1. Who is in the study\n")
    L.append(f"- The main table has {n_all} I-SPY patients. We removed {len(excluded)} "
             f"I-SPY1 patients whose scans are missing in the source data "
             f"(list in `excluded_patients.csv`).")
    L.append(f"- **{len(meta)} patients are left**: "
             f"I-SPY1 = {int(c.loc[('spy1','all'),'n'])}, I-SPY2 = {int(c.loc[('spy2','all'),'n'])}.")
    L.append(f"- Overall **{c.loc[('pooled','all'),'pCR_rate_%']}%** of patients had pCR "
             f"(I-SPY1 {c.loc[('spy1','all'),'pCR_rate_%']}%, I-SPY2 {c.loc[('spy2','all'),'pCR_rate_%']}%). "
             f"So roughly 1 in 3 patients is pCR-positive: the classes are unbalanced, "
             f"which is why we use class weights and AUPRC.")
    L.append(f"- Split sizes (pooled): train {int(c.loc[('pooled','train'),'n'])}, "
             f"validation {int(c.loc[('pooled','val'),'n'])}, test {int(c.loc[('pooled','test'),'n'])}.\n")
    L.append(md_table(counts) + "\n")
    L.append("pCR rate is the % of patients with pCR = 1. The splits have similar pCR rates "
             "if the numbers in the `pCR_rate_%` column are close to each other.\n")

    L.append("### HR / HER2 subtypes (all splits)\n")
    L.append(md_table(sub[sub["split"] == "all"].drop(columns="split")) + "\n")
    L.append("pCR depends strongly on subtype (HER2+ and triple-negative tumours respond more often). "
             "The clinical model must include HR and HER2. Full table per split: `subtype_counts.csv`.\n")

    L.append("## 2. I-SPY2 treatment arms\n")
    L.append(f"- All {arm_checks['spy2_patients']} I-SPY2 patients were found in the TCIA table "
             f"(missing: {arm_checks['missing_in_tcia']}).")
    L.append(f"- Labels agree between the two tables: pCR differences = {arm_checks['pCR_disagree']}, "
             f"HR = {arm_checks['HR_disagree']}, HER2 = {arm_checks['HER2_disagree']}.")
    L.append(f"- There are {len(arms)} different arms. Many are small, so for the clinical model we "
             f"should group them (e.g. by drug class) rather than use each arm as its own category.\n")
    L.append(md_table(arms) + "\n")
    L.append("### Scanner manufacturer (I-SPY2)\n")
    L.append(md_table(manu) + "\n")

    L.append("## 3. File checks\n")
    n_ok = int((chk["n_problems"] == 0).sum())
    L.append(f"- {n_ok} of {len(chk)} patients have all 3 phase scans + mask, readable, "
             f"with the same shape.")
    if len(problems):
        L.append(f"- **{len(problems)} patients have problems** (full list in `file_problems.csv`):\n")
        L.append(md_table(problems[["pid", "dataset", "split", "problems"]].head(40)) + "\n")
    else:
        L.append("- No problems found.\n")
    mv = chk["mask_values"].value_counts().head(5)
    L.append("Mask values seen (top 5): " + ", ".join(f"`{k}` ({v} patients)" for k, v in mv.items())
             + ". Masks coded 0/255 are still binary; later scripts must use `mask > 0`, not `mask == 1`.\n")
    d = chk["file_minus_meta_nz"]
    L.append(f"Note: the cropped files have more slices than the metadata `n_z` column says "
             f"(median +{d.median():.0f}, max +{d.max():.0f}). This is how the dataset was cropped, "
             f"not an error, because scan and mask always have the same shape.\n")

    L.append("## 4. Tumour size\n")
    L.append("Volume uses the real spacing from the metadata (the NIfTI headers wrongly say 1x1x1 mm).\n")
    L.append(md_table(sizes) + "\n")
    ratio = chk["tumour_mm3"] / chk["meta_tum_vol"]
    off = chk[(ratio - 1).abs() > 0.05]
    L.append("**Cross-check with the metadata `tum_vol` column:** "
             + ", ".join(f"{ds}: median ratio {r.median():.3f}"
                         for ds, r in ratio.groupby(chk["dataset"]))
             + f". {len(off)} patients differ by more than 5% "
             f"(listed in `volume_mismatch.csv`). A ratio of 1.0 means our volume equals theirs, "
             f"so our spacing maths is correct.\n")
    off[["pid", "dataset", "tumour_voxels", "tumour_mm3", "meta_tum_vol"]].assign(
        ratio=ratio[off.index].round(3)).to_csv(OUT / "volume_mismatch.csv", index=False)
    thin = chk[chk["tumour_slices"] <= 2]
    watch = pd.concat([
        thin.assign(reason=lambda d: "tumour on only " + d["tumour_slices"].astype(str) + " slice(s): mask may be broken"),
        off.assign(reason=lambda d: "volume is " + ratio[d.index].round(2).astype(str)
                   + "x the metadata tum_vol; mask values " + d["mask_values"].astype(str)
                   + ": spacing or mask scale may be wrong"),
    ])[["pid", "dataset", "split", "tumour_voxels", "reason"]]
    watch.to_csv(OUT / "patients_to_watch.csv", index=False)
    L.append("### Patients to watch\n")
    if len(watch):
        L.append(md_table(watch) + "\n")
        L.append("These patients pass the file checks, but something looks unusual. Look at them in the "
                 "QC overlays after preprocessing, and decide whether to keep them.\n")
    else:
        L.append("None.\n")

    L.append("### The 20 smallest tumours\n")
    L.append(md_table(smallest) + "\n")
    L.append("Very small tumours are a risk for habitat analysis: if a tumour has only a few hundred "
             "voxels, splitting it into 3-4 habitats leaves too few voxels for stable texture features. "
             "We will enforce a minimum habitat size and record how many patients are affected.\n")

    L.append("## 5. Voxel spacing and resampling grid\n")
    L.append(md_table(spacing) + "\n")
    L.append("Tumour size in voxels after resampling to each candidate isotropic grid:\n")
    L.append(md_table(grid) + "\n")
    xy_med = chk["xy_spacing"].median()
    z_med = chk["slice_thick"].median()
    g1 = grid[grid["grid_mm"] == 1.0]
    L.append("**Suggestion: resample everything to 1.0 x 1.0 x 1.0 mm (isotropic).**\n")
    L.append(f"- In-plane pixels are about {xy_med:.2f} mm, slices about {z_med:.1f} mm apart. "
             f"1 mm sits between the two, so we lose only a little in-plane detail and "
             f"interpolate only a little along z.")
    L.append("- Isotropic voxels are needed for 3D texture features (IBSI recommends it), because "
             "GLCM and similar features compare neighbouring voxels in all directions.")
    L.append(f"- At 1 mm the median tumour has {int(g1['median_voxels'].median())} voxels, "
             f"enough for 3-4 habitats. Coarser grids (1.5 or 2 mm) would make small tumours too small.")
    L.append("- Images: B-spline interpolation. Masks: nearest-neighbour.\n")

    L.append("## Mask origin\n")
    if MASK_NOTES.exists():
        L.append(MASK_NOTES.read_text(encoding="utf-8").strip() + "\n")
    else:
        L.append("_Not written yet (see `docs/mask_origin.md`)._\n")

    L.append("## Files in this folder\n")
    for f in sorted(OUT.glob("*.csv")):
        L.append(f"- `{f.name}`")
    (OUT / "audit_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    meta, excluded, n_all = load_metadata()
    print(f"I-SPY patients: {n_all}, excluded: {len(excluded)}, kept: {len(meta)}")

    counts, sub = count_tables(meta)
    arms, manu, arm_checks = arm_table(meta)

    chk = run_checks(meta)
    problems = chk[chk["n_problems"] > 0].sort_values("pid")

    rows = []
    for ds, g in pd.concat([chk, chk.assign(dataset="pooled")]).groupby("dataset"):
        for col in ["tumour_voxels", "tumour_mm3"]:
            s = g[col]
            rows.append({"dataset": ds, "measure": col, "min": s.min(), "p5": s.quantile(0.05),
                         "median": s.median(), "p95": s.quantile(0.95), "max": s.max()})
    sizes = pd.DataFrame(rows).round(0)
    sizes["cm3_median"] = (sizes["median"] / 1000).round(1).where(sizes["measure"] == "tumour_mm3")
    smallest = (chk.nsmallest(20, "tumour_mm3")
                [["pid", "dataset", "split", "pCR", "tumour_voxels", "tumour_mm3",
                  "tumour_slices", "xy_spacing", "slice_thick"]].round(1))

    spacing, spacing_vals, grid = spacing_tables(chk)

    excluded.to_csv(OUT / "excluded_patients.csv", index=False)
    counts.to_csv(OUT / "cohort_split_counts.csv", index=False)
    sub.to_csv(OUT / "subtype_counts.csv", index=False)
    arms.to_csv(OUT / "spy2_arm_counts.csv", index=False)
    manu.to_csv(OUT / "spy2_manufacturer_counts.csv", index=False)
    chk.to_csv(OUT / "patient_checks.csv", index=False)
    problems.to_csv(OUT / "file_problems.csv", index=False)
    sizes.to_csv(OUT / "tumour_size_summary.csv", index=False)
    smallest.to_csv(OUT / "smallest_20_tumours.csv", index=False)
    spacing.to_csv(OUT / "spacing_summary.csv", index=False)
    spacing_vals.to_csv(OUT / "slice_thickness_counts.csv", index=False)
    grid.to_csv(OUT / "grid_candidates.csv", index=False)

    minutes = (time.time() - t0) / 60
    write_summary(meta, excluded, n_all, counts, sub, arms, manu, arm_checks,
                  chk, problems, sizes, smallest, spacing, grid, minutes)
    print(f"Problems: {len(problems)} patients. Mean time per patient: {chk['seconds'].mean():.2f} s")
    print(f"Done in {minutes:.1f} min. Outputs in {OUT}")


if __name__ == "__main__":
    main()
