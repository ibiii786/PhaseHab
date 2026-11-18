# PhaseHab work log

One entry per step. Newest at the bottom. Dates are YYYY-MM-DD.

---

## 2026-11-18 — Setup

- Created `code\.venv` with Python 3.12.6 and installed `requirements.txt`.
- PyRadiomics 3.1.1.dev installed from GitHub (built without errors, no compiler fix needed).
- Import test: radiomics, SimpleITK, nibabel, xgboost, shap all OK.
- Connected `code\` to GitHub (`ibiii786/PhaseHab`, branch `main`). Rule: push after every commit.

## 2026-11-18 — Step 1: cohort audit (`src\01_audit.py`)

- 1,154 I-SPY patients in the main table; 14 I-SPY1 excluded (scans missing) → **1,140** (I-SPY1 158, I-SPY2 982).
- Splits: train 881, val 131, test 128. pCR rate 31.6% overall, 31–33% in every split.
- pCR by subtype: HER2+ 46%, triple-negative 39%, HR+/HER2- 16%.
- I-SPY2: 13 treatment arms (pCR 17–59%); TCIA table labels agree 100% with the main table.
- All 1,140 patients have 3 phases + mask with matching shapes. Tumour volume matches metadata `tum_vol`.
- Median tumour 14.7 cm³ (0.2–495). Median spacing 0.70 mm in-plane, 2.0 mm slices.
- Decision: resample to **1 mm isotropic**.
- Mask origin (docs + own check on 50 patients): masks are FTV masks defined by **early-phase enhancement ≥ ~70%**
  → known limitation. Notes in `docs\mask_origin.md`.
- Watch list: ISPY2-553012 (1 slice), ISPY2-726091 and ISPY2-934459 (0/255 masks, volume 4× metadata).
- Outputs: `outputs\audit\` (CSV tables + `audit_summary.md`).

## 2026-11-18 — Part A: analysis plan (`config\analysis_plan.md`)

- Fixed all main decisions before seeing any results: cohort, splits, resampling, normalisation, habitats,
  feature blocks, models, statistics, limitations, sensitivity analyses.
- Habitat K-means is fitted once on train + val voxels (label-free, test never used), with a per-fold ARI check.
- Proposed primary classifier for the phase tests: L2 logistic regression after LASSO.
- Test-set minimum detectable AUROC difference ≈ 0.15 (α 0.05), ≈ 0.17 (Holm α 0.0125).
