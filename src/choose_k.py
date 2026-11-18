r"""
choose_k.py - Compare the number of habitats k = 2, 3, 4 on CROSS-VALIDATION results only (never the test set).

Rule (written in config/analysis_plan.md on 2026-11-18, BEFORE the k = 2 and k = 4 results existed):
  1. Eligible k = passes the habitat stability rule (minimum per-fold ARI >= 0.90): k = 2 and k = 3.
     k = 4 (min ARI 0.898) would need per-fold K-means, which is too slow, so it is reported as exploratory only.
  2. Score = mean pooled CV AUROC (primary LR) of the 4 imaging models: pre, early, late, fused.
  3. k = 3 stays the main k unless another eligible k scores at least 0.01 higher.

Inputs : outputs/models/k{k}/cv_metrics.csv and cv_differences.csv, outputs/habitats/stability_ari.csv and
         small_habitats_summary.csv
Outputs: outputs/models/k_selection.csv and outputs/models/k_selection.md (plain-English summary)
Run from code\ :  .venv\Scripts\python.exe src\choose_k.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paths as P  # noqa: E402

KS = [2, 3, 4]
DEFAULT_K = 3
MARGIN = 0.01
MIN_ARI = 0.90
IMAGING = ["pre", "early", "late", "fused"]
ALL_MODELS = ["clinical", "pre", "early", "late", "fused", "clinical_best"]


def main():
    mdir = P.OUT / "models"
    stab = pd.read_csv(P.OUT / "habitats" / "stability_ari.csv").groupby("k")["ARI"].min()
    small = pd.read_csv(P.OUT / "habitats" / "small_habitats_summary.csv").set_index("k")

    rows, diffs = [], []
    for k in KS:
        f = mdir / f"k{k}" / "cv_metrics.csv"
        if not (mdir / f"k{k}" / "DONE.txt").exists() or not f.exists():
            print(f"k = {k}: no finished results yet ({f.parent}) - skipped")
            continue
        m = pd.read_csv(f)
        m = m[m.cohort == "pooled"]
        row = {"k": k, "min_ARI": stab.get(k), "eligible": bool(stab.get(k, 0) >= MIN_ARI),
               "pct_small_habitat": small.loc[k, "%_patients"]}
        for clf in ["lr", "xgb"]:
            mm = m[m.classifier == clf].set_index("model")
            for mod in ALL_MODELS:
                if mod in mm.index:
                    row[f"{mod}_{clf}"] = mm.loc[mod, "AUROC_mean"]
                    row[f"{mod}_{clf}_sd"] = mm.loc[mod, "AUROC_sd"]
            row[f"score_{clf}"] = mm.loc[IMAGING, "AUROC_mean"].mean()
        phases = {p: row[f"{p}_lr"] for p in ["pre", "early", "late"]}
        row["best_phase_lr"] = max(phases, key=phases.get)
        rows.append(row)
        d = pd.read_csv(mdir / f"k{k}" / "cv_differences.csv")
        diffs.append(d.assign(k=k))
    if not rows:
        print("No results found.")
        return
    tab = pd.DataFrame(rows)

    elig = tab[tab.eligible]
    base = tab.loc[tab.k == DEFAULT_K, "score_lr"]
    chosen, reason = DEFAULT_K, "k = 3 kept (default)"
    if len(base) and len(elig):
        best = elig.sort_values("score_lr", ascending=False).iloc[0]
        gain = best["score_lr"] - base.iloc[0]
        if best["k"] != DEFAULT_K and gain >= MARGIN:
            chosen = int(best["k"])
            reason = f"k = {chosen} beats k = 3 by {gain:.3f} (>= {MARGIN}) on the mean imaging AUROC"
        elif best["k"] != DEFAULT_K:
            reason = (f"k = {int(best['k'])} is only {gain:+.3f} better than k = 3 (< {MARGIN}), so k = 3 is kept")
        else:
            reason = "k = 3 has the best score among eligible k"
    tab["chosen"] = tab["k"] == chosen
    tab.round(4).to_csv(mdir / "k_selection.csv", index=False)
    dd = pd.concat(diffs, ignore_index=True)
    dd.to_csv(mdir / "k_selection_phase_differences.csv", index=False)

    # plain-English report
    lines = ["# Choice of k (number of habitats) - cross-validation only, test set not used", "",
             "Rule (fixed before the k = 2 / k = 4 results): eligible k must pass the stability check (min ARI >= 0.90);",
             "score = mean pooled CV AUROC of the pre, early, late and fused LR models; k = 3 stays unless another",
             f"eligible k is at least {MARGIN} better.", "",
             "| k | min ARI | eligible | % patients with a habitat < 100 voxels | pre | early | late | fused | "
             "clinical + best | score (LR) | score (XGB) | best phase |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        lines.append(f"| {r.k} | {r.min_ARI:.3f} | {'yes' if r.eligible else 'no (exploratory)'} | "
                     f"{r.pct_small_habitat:.0f}% | {r.pre_lr:.3f} | {r.early_lr:.3f} | {r.late_lr:.3f} | "
                     f"{r.fused_lr:.3f} | {r.clinical_best_lr:.3f} | **{r.score_lr:.3f}** | {r.score_xgb:.3f} | "
                     f"{r.best_phase_lr} |")
    lines += ["", f"**Chosen k = {chosen}.** {reason}.", "",
              "Phase differences (pooled AUROC, LR, mean over 5 repeats) for each k:", ""]
    sub = dd[(dd.classifier == "lr") & dd.comparison.isin(["early - pre", "early - late", "late - pre"])]
    lines.append(sub.pivot_table(index="comparison", columns="k", values="AUROC_diff_mean").round(3).to_markdown())
    (mdir / "k_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
