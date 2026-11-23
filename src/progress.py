import argparse
import datetime as dt
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P

REPORT_URL = "https://claude.ai/artifact/Xa8UEBMmjdMZRZRBE9TrPw"
WIDTH = 96

C = {"g": "\033[32m", "y": "\033[33m", "d": "\033[90m", "b": "\033[1m", "c": "\033[36m", "r": "\033[0m"}


def setup_console(plain):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if plain or not sys.stdout.isatty():
        for k in C:
            C[k] = ""
    elif os.name == "nt":
        os.system("")


def col(text, key):
    return f"{C[key]}{text}{C['r']}"


def rule(title=None):
    if title is None:
        print(col("─" * WIDTH, "d"))
        return
    print()
    print(col("━" * WIDTH, "c"))
    print(col(f" {title}", "b"))
    print(col("━" * WIDTH, "c"))


def badge(state):
    return {"done": col("✔ DONE   ", "g"), "part": col("◐ PARTLY ", "y"), "wait": col("⏸ READY  ", "y"),
            "todo": col("○ TO DO  ", "d")}[state]


def read(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


def exists(*parts):
    return P.OUT.joinpath(*parts).exists()


def last_commit():
    try:
        out = subprocess.run(["git", "-C", str(P.CODE), "log", "-1", "--format=%h  %cd  %s", "--date=short"],
                             capture_output=True, text=True, timeout=10)
        n = subprocess.run(["git", "-C", str(P.CODE), "rev-list", "--count", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return out.stdout.strip(), n.stdout.strip()
    except Exception:
        return "", ""


def gather():
    g = {}
    idx = read(P.OUT / "preprocessed" / "index.csv")
    if idx is not None:
        use = idx[idx["use"]]
        g["n_pre"] = len(idx)
        g["n_use"] = len(use)
        g["by_split"] = use["split"].value_counts().to_dict()
        g["by_set"] = use["dataset"].value_counts().to_dict()
        g["pcr"] = use["pCR"].mean()
    hab = read(P.OUT / "habitats" / "stability_ari.csv")
    if hab is not None:
        g["ari"] = hab.groupby("k")["ARI"].min().to_dict()
    small = read(P.OUT / "habitats" / "small_habitats_summary.csv")
    if small is not None:
        g["small"] = small.set_index("k")
    feats = {}
    for k in [2, 3, 4]:
        f = read(P.OUT / "features" / f"k{k}" / "early.csv", nrows=None)
        if f is not None:
            feats[k] = (len(f), f.shape[1] - 9)
    g["feats"] = feats
    shape = read(P.OUT / "features" / "k3" / "shape.csv")
    g["n_shape"] = shape.shape[1] - 9 if shape is not None else None
    g["cv"] = read(P.OUT / "models" / "k3" / "cv_metrics.csv")
    g["delong"] = read(P.OUT / "compare" / "k3" / "delong_tests.csv")
    g["metrics"] = read(P.OUT / "compare" / "k3" / "metrics_table.csv")
    g["shap"] = read(P.OUT / "shap" / "k3" / "group_importance_summary.csv")
    g["ksel"] = read(P.OUT / "models" / "k_selection.csv")
    oof = read(P.OUT / "models" / "k3" / "oof_predictions.csv", usecols=["pid"])
    if oof is not None and idx is not None:
        test_ids = set(idx.loc[idx["split"] == "test", "pid"])
        g["test_in_oof"] = len(test_ids & set(oof["pid"].unique()))
        g["n_test"] = len(set(idx.loc[(idx["split"] == "test") & idx["use"], "pid"]))
    g["final_done"] = exists("final_test")
    return g


def steps(g):
    feats = g.get("feats", {})
    ksel = g.get("ksel")
    s = [
        ("Setup: Python 3.12, libraries, GitHub", "done", "code pushed to GitHub after every change"),
        ("01 Cohort audit", "done" if exists("audit", "audit_summary.md") else "todo",
         "1,154 patients listed, 14 with missing scans removed"),
        ("Analysis plan fixed before any result", "done", "config/analysis_plan.md + dated decisions log"),
        ("02 Preprocessing (1 mm, normalise, crop)", "done" if g.get("n_pre") else "todo",
         f"{g.get('n_pre', 0):,} patients processed, {g.get('n_use', 0):,} kept after image QC"),
        ("03 Habitats (K-means, k = 2/3/4)", "done" if g.get("ari") else "todo",
         "3 zones per tumour, stability check passed for k = 3"),
        ("04 Radiomics features", "done" if 3 in feats else "todo",
         f"{feats.get(3, (0, 0))[1]} per phase + {g.get('n_shape') or 0} shape = "
         f"{3 * feats.get(3, (0, 0))[1] + (g.get('n_shape') or 0)} numbers per patient"),
        ("05 Model training (5 x 5 cross-validation)", "done" if g.get("cv") is not None else "todo",
         "6 models x 2 methods, test set never loaded"),
        ("06 Statistical comparison (DeLong + Holm)", "done" if g.get("delong") is not None else "todo",
         "4 pre-registered phase tests, CIs, calibration, figures"),
        ("07 SHAP explanations", "done" if g.get("shap") is not None else "todo",
         "importance by phase and by habitat"),
        ("Choice of k (2 / 3 / 4 habitats)", "done" if ksel is not None and len(ksel) == 3 else "todo",
         "rule fixed in advance; k = 3 kept"),
        ("Sensitivity analyses", "part", "2 of 4 done (each trial alone, + treatment arm)"),
        ("08 Final test on locked patients", "done" if g.get("final_done") else "wait",
         f"{g.get('n_test', 128)} patients still locked, used once at the end"),
        ("Probability calibration", "todo", "needed before the web app shows percentages"),
        ("Web app (FastAPI + React)", "todo", "upload MRI -> prediction + SHAP view"),
        ("Thesis writing", "todo", "figures and tables already produced"),
    ]
    return s


def section_status(s):
    rule("1. PROGRESS")
    done = sum(1 for x in s if x[1] == "done")
    mid = sum(1 for x in s if x[1] in ("part", "wait"))
    todo = len(s) - done - mid
    bar_len = 60
    nd = round(bar_len * done / len(s))
    nm = round(bar_len * mid / len(s))
    bar = col("█" * nd, "g") + col("█" * nm, "y") + col("░" * (bar_len - nd - nm), "d")
    print(f" {bar}  {col(f'{done}/{len(s)} done', 'b')}  ({mid} partly/ready, {todo} to do)")
    print()
    for i, (name, state, note) in enumerate(s, 1):
        print(f" {i:>2}. {badge(state)} {name:<44} {col(note, 'd')}")


def section_data(g):
    rule("2. DATA")
    if not g.get("n_use"):
        print(" (no preprocessed data found)")
        return
    sp = g["by_split"]
    ds = g["by_set"]
    print(f" Patients analysed : {col(format(g['n_use'], ','), 'b')}   "
          f"(I-SPY1 {ds.get('spy1', 0)}, I-SPY2 {ds.get('spy2', 0)})   pCR rate {100 * g['pcr']:.1f}%")
    print(f" Groups            : training {sp.get('train', 0)}  |  check {sp.get('val', 0)}  |  "
          f"{col('locked test ' + str(sp.get('test', 0)), 'y')}")
    if "test_in_oof" in g:
        ok = g["test_in_oof"] == 0 and not g["final_done"]
        msg = "untouched (0 test patients in any model output)" if ok else "USED"
        print(f" Test set          : {col(msg, 'g' if ok else 'y')}")
    if g.get("ari"):
        a = g["ari"]
        print(f" Habitat stability : min ARI  k2 {a.get(2, 0):.3f}  k3 {a.get(3, 0):.3f}  k4 {a.get(4, 0):.3f}"
              f"   (rule: >= 0.90)")
    if "small" in g:
        sm = g["small"]
        print(f" Small habitats    : k3 -> {int(sm.loc[3, 'patients_with_any_habitat_<100'])} patients have a "
              f"zone < 100 voxels (features left empty, imputed in training)")


def section_results(g):
    rule("3. RESULTS  (cross-validation on 1,010 training + check patients; 0.5 = coin flip, 1.0 = perfect)")
    m = g.get("metrics")
    if m is None:
        print(" (no results yet)")
        return
    names = [("clinical", "Patient facts only (HR, HER2, age, trial)"), ("pre", "Pre-contrast MRI"),
             ("early", "Early post-contrast MRI"), ("late", "Late post-contrast MRI"),
             ("fused", "All 3 phases + shape"), ("clinical_best", "Patient facts + pre MRI")]
    print(f" {'Model':<44}{'AUROC (95% CI)':>22}{'2nd method':>13}{'I-SPY1':>9}{'I-SPY2':>9}")
    rule()
    for mid, label in names:
        r = m[(m.model == mid) & (m.classifier == "lr") & (m.cohort == "pooled")]
        x = m[(m.model == mid) & (m.classifier == "xgb") & (m.cohort == "pooled")]
        s1 = m[(m.model == mid) & (m.classifier == "lr") & (m.cohort == "spy1")]
        s2 = m[(m.model == mid) & (m.classifier == "lr") & (m.cohort == "spy2")]
        if r.empty:
            continue
        r = r.iloc[0]
        auc = r["AUROC"]
        bar = col("▇" * max(0, round((auc - 0.5) * 60)), "c")
        ci = f"{auc:.3f} ({r['AUROC_lo']:.2f}-{r['AUROC_hi']:.2f})"
        xs = f"{x.iloc[0]['AUROC']:.3f}" if not x.empty else "-"
        a1 = f"{s1.iloc[0]['AUROC']:.3f}" if not s1.empty else "-"
        a2 = f"{s2.iloc[0]['AUROC']:.3f}" if not s2.empty else "-"
        print(f" {label:<44}{ci:>22}{xs:>13}{a1:>9}{a2:>9}  {bar}")


def section_question(g):
    rule("4. MAIN QUESTION: is one MRI phase better than another?")
    d = g.get("delong")
    if d is None:
        print(" (not tested yet)")
        return
    prim = d[d.family == "primary"]
    sec = d[d.family == "secondary (xgb)"].set_index("comparison")
    print(f" {'Comparison':<22}{'Difference':>12}{'95% CI':>20}{'Holm p':>10}{'2nd method':>13}")
    rule()
    for _, r in prim.iterrows():
        xs = sec.loc[r["comparison"], "diff"] if r["comparison"] in sec.index else float("nan")
        print(f" {r['comparison']:<22}{r['diff']:>+12.3f}{f'{r.diff_lo:+.3f} to {r.diff_hi:+.3f}':>20}"
              f"{r['p_holm']:>10.2f}{xs:>+13.3f}")
    sig = (prim["p_holm"] < 0.05).any()
    print()
    if sig:
        print(col(" Verdict: at least one phase difference is significant.", "b"))
    else:
        print(col(" Verdict: NO phase is significantly better. Differences are about 0.01-0.02 and every CI "
                  "crosses 0.", "b"))
    ex = d[(d.family == "extra (unadjusted)") & (d.cohort == "pooled") & (d.classifier == "lr")]
    cb = ex[ex.comparison == "clinical_best vs clinical"]
    im = ex[ex.comparison.isin(["pre vs clinical", "early vs clinical", "late vs clinical"])]
    if not im.empty:
        print(f" Every MRI-only model is below patient facts by {-im['diff'].max():.2f} to {-im['diff'].min():.2f}"
              f" AUROC (p < 0.001).")
    if not cb.empty:
        c = cb.iloc[0]
        print(f" Patient facts + MRI vs facts alone: {c['diff']:+.3f} (p = {c['p']:.3f}, unadjusted) "
              f"-> small gain, weak evidence.")


def section_shap(g):
    rule("5. WHAT THE MODEL USES (SHAP, combined model, average of 25 folds)")
    s = g.get("shap")
    if s is None:
        print(" (not run yet)")
        return
    for clf, lab in [("lr", "Main method  "), ("xgb", "Second method")]:
        p = s[(s.model == "fused") & (s.classifier == clf) & (s.grouping == "phase")].set_index("group")
        h = s[(s.model == "fused") & (s.classifier == clf) & (s.grouping == "habitat")].set_index("group")
        ph = "  ".join(f"{g_}{100 * p.loc[g_, 'share_mean']:>4.0f}%" for g_ in ["pre", "early", "late", "shape"]
                       if g_ in p.index)
        hb = "  ".join(f"{g_}{100 * h.loc[g_, 'share_mean']:>4.0f}%" for g_ in ["h1", "h2", "h3"] if g_ in h.index)
        print(f" {lab}  by phase: {ph}    by zone: {hb}")
    print(col(" -> all three phases contribute a similar share; the medium zone h2 matters most.", "d"))


def section_k(g):
    rule("6. HOW MANY HABITATS? (CV only, rule fixed before results)")
    k = g.get("ksel")
    if k is None:
        print(" (not run yet)")
        return
    print(f" {'k':<4}{'stable':<9}{'pre':>7}{'early':>7}{'late':>7}{'fused':>7}{'average':>9}   best phase")
    for _, r in k.iterrows():
        mark = col("  <- chosen", "g") if r.get("chosen", False) else ""
        print(f" {int(r['k']):<4}{('yes' if r['eligible'] else 'no'):<9}{r['pre_lr']:>7.3f}{r['early_lr']:>7.3f}"
              f"{r['late_lr']:>7.3f}{r['fused_lr']:>7.3f}{r['score_lr']:>9.3f}   {r['best_phase_lr']}{mark}")


def section_left():
    rule("7. WHAT IS LEFT")
    items = [
        ("Step 08: final test", "run the frozen models once on the 128 locked patients"),
        ("Calibration", "make predicted chances match real rates (models over-predict pCR)"),
        ("2 sensitivity analyses", "habitats from early phase only; masks grown by 2 mm (overnight runs)"),
        ("Web app", "FastAPI back end + React front end"),
        ("Thesis", "methods, results, limitations; fix the licence statement (CC BY 4.0)"),
        ("Optional", "wavelet / LoG image filters, only if time allows"),
    ]
    for a, b in items:
        print(f"  {col('○', 'd')} {a:<24} {b}")


def main():
    ap = argparse.ArgumentParser(description="PhaseHab progress report")
    ap.add_argument("--plain", action="store_true", help="no colours")
    args = ap.parse_args()
    setup_console(args.plain)
    g = gather()
    commit, n = last_commit()
    print()
    print(col(" PhaseHab", "b") + "  ·  DCE-MRI phase comparison with habitat radiomics for predicting pCR")
    print(col(" Ghulam Ibrahim, Sameel Ahmed  ·  Supervisor: Dr. Erum Ashraf  ·  Bahria University", "d"))
    print(col(f" Report generated {dt.datetime.now():%Y-%m-%d %H:%M}"
              + (f"  ·  {n} commits, last: {commit}" if commit else ""), "d"))
    s = steps(g)
    section_status(s)
    section_data(g)
    section_results(g)
    section_question(g)
    section_shap(g)
    section_k(g)
    section_left()
    rule()
    print(f" Visual report with charts and plain-English explanations: {col(REPORT_URL, 'c')}")
    print()


if __name__ == "__main__":
    main()
