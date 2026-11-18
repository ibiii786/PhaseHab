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

## 2026-11-18 — Part B: preprocessing (`src\02_preprocess.py`)

- Resample to 1 mm (B-spline / nearest neighbour), Otsu + pre-contrast normalisation for all 3 phases,
  crop to tumour + 10 mm. Output: `outputs\preprocessed\<pid>.npz` + `index.csv`.
- Pilot (50 training patients): 0 errors; ~19 s per patient on one worker.
- Full run (1,140 patients, 4 workers; PC has 2 cores / 4 threads): 0 errors, 81 min, 6.9 GB.
  Tumour volume kept after resampling (median ratio 1.000, range 0.87–1.07; the extremes are tiny tumours).
- QC (20 random + 3 watch patients, `outputs\qc\preprocess\`): masks on the enhancing tissue, phases aligned,
  I-SPY1 sagittal geometry correct.
- Decisions: ISPY2-553012 excluded; ISPY2-726091 and ISPY2-934459 kept (see plan decisions log).
- Problems found: ISPY1_1156 (metadata "early" is a second pre-contrast scan) and ISPY1_1187 (saturated
  pre-contrast voxels break normalisation). Decision pending.
- Noted: many FTV masks contain scattered small specks outside the main tumour (median 40 pieces).

## 2026-11-18 — Fix 2 problem patients

- ISPY1_1156: phase choice corrected to acq 0/2/3 (`paths.PHASE_OVERRIDE`), re-processed, QC checked: normal enhancement.
- ISPY1_1187: excluded (saturated pre-contrast image).
- New `--only <pids>` option in `02_preprocess.py` to re-run single patients and update `index.csv`.
- Analysis cohort: **1,138 patients**.

## 2026-11-18 — Step 03: habitats (`src\03_habitats.py`)

- One K-means per k (2, 3, 4) on 2.0 M voxels (≤ 2,000 per patient) from 1,010 train + val patients; test never used.
  All 1,138 patients assigned. Maps in `outputs\habitats\k{k}\`, tables `habitat_table_k{k}.csv`, centres `centres.csv`.
- Switched MiniBatchKMeans → full K-means (MiniBatch was unstable between seeds). Logged in the plan.
- k=3 centres (pre / early / late, normalised): h1 −0.03 / 2.98 / 3.16 low enhancement (39% of voxels);
  h2 1.01 / 6.03 / 6.18 medium (47%); h3 2.02 / 10.32 / 10.01 high with slight wash-out (14%). Silhouette 0.41.
- Stability (min per-fold ARI): k2 0.97, k3 0.91 (pass), k4 0.90 (just fails → per-fold fitting if k4 is used).
- Habitat < 100 voxels for k=3: 389 patients (34%), mostly h3 (380).
- QC (`outputs\qc\habitats\`): high-enhancement core, low-enhancement rim. Weakly enhancing tumours can be all h1.
- Also added `src\cv.py` (shared cohort + CV folds, so all scripts use the same folds).

## 2026-11-18 — Step 04: radiomics features (`src\04_features.py`, `config\radiomics.yaml`) — pilot

- `config\radiomics.yaml`: Original image only, 3D, no resampling, binWidth **0.1** (normalised units), same for all
  phases. Chosen on training patients: 90% of habitat regions get 30–130 bins (`outputs\features\k3\bin_check.csv`).
- Per patient × phase × habitat (k = 3): firstorder, glcm, glrlm, glszm, gldm, ngtdm → **279 features per phase**.
  Habitat < 100 voxels → NaN. Shape once per patient (14) + habitat volume fractions (3).
- Pilot, 20 training patients: 0 errors, ~10 s per patient per worker, ~3 s per patient overall (4 workers)
  → full run estimated at ~1 hour.
- Each patient is saved to `parts\<pid>.json` at once; CSVs rebuilt every 100 patients; restart skips finished patients.
- Full run: `code\run_features.bat` (double-click), started by the user outside Claude Code.

## 2026-11-18 — Step 04: full feature run (k = 3)

- Run by the user with `run_features.bat`: 1,118 patients in 83 min (+ 20 from the pilot), **0 errors**, no inf values.
  Slowest patients (large tumours) took up to ~5 min each per worker.
- `outputs\features\k3\`: pre / early / late .csv (1,138 rows × 279 features each), shape.csv (17 features).
  Splits: 879 train, 131 val, 128 test. pCR rate 0.316.
- Empty (NaN) habitat blocks: h1 10 patients, h2 42, h3 380 → 12.7% of phase feature values are NaN (imputed in folds).
- Quick look (train + val, single features): the strongest features only reach AUROC ≈ 0.59 (e.g. early_h2_ngtdm_Strength),
  so imaging signal per feature is weak.

## 2026-11-18 — Step 05: training + cross-validation (`src\05_train.py`)

- Development set = train + val (1,010 patients, pCR 31.6%). Test set not loaded. 5 × 5 repeated stratified CV
  (`cv.dev_folds`). All learning steps inside an sklearn Pipeline per fold. Run time 8.6 min.
- Pooled CV AUROC, mean ± SD over 5 repeats (primary = LASSO → L2 logistic):
  clinical 0.673 ± 0.007 | pre 0.566 ± 0.005 | early 0.553 ± 0.016 | late 0.566 ± 0.007 | fused 0.573 ± 0.017 |
  clinical + pre 0.686 ± 0.009. XGBoost: clinical 0.655, pre 0.550, early 0.557, late 0.537, fused 0.563.
- Phase differences are tiny (≤ 0.013 AUROC) and change direction between classifiers → no phase stands out.
  Every imaging model is ~0.11 below the clinical model. Adding pre-phase features to clinical: +0.013.
- By cohort: imaging models work a little on I-SPY2 (0.56–0.59) but not on I-SPY1 (0.46–0.53).
- Outputs in `outputs\models\`: oof_predictions.csv, cv_metrics.csv, cv_metrics_by_repeat.csv, cv_differences.csv,
  selected_features_*.csv, 13 refit models `<model>_k3_<lr|xgb>.joblib`, run_05_train.log.

## 2026-11-18 — Step 06: statistical comparison (`src\06_compare.py`, `src\stats.py`)

- Repeat-averaged OOF predictions, development set only (1,010). Test set not used.
- **Primary result (lr, pooled, Holm over 4 tests): no significant phase difference.**
  early − pre −0.019 (95% CI −0.056 to 0.018), early − late −0.012, late − pre −0.006, fused − pre +0.010;
  all Holm p = 1.00. XGBoost: differences ≤ 0.024, all Holm p ≥ 0.75. Same picture in I-SPY1 and I-SPY2 alone.
- AUROC (95% CI), lr pooled: clinical 0.675 (0.642–0.709), pre 0.578 (0.537–0.615), early 0.559 (0.520–0.598),
  late 0.571 (0.532–0.610), fused 0.587 (0.545–0.626), clinical + pre 0.699 (0.665–0.732).
- Every imaging model is significantly worse than clinical (−0.09 to −0.12, unadjusted p < 0.001).
  Clinical + pre beats clinical by +0.024 (p = 0.044, unadjusted; lr only, not with xgb).
- I-SPY1: imaging models at chance (0.45–0.50); clinical 0.74. I-SPY2: imaging 0.57–0.60; clinical 0.66.
- Calibration: balanced class weights → probabilities too high (≈ +0.17 on average). Brier of imaging models
  (0.24–0.25) is worse than the no-skill value 0.216.
- Outputs: `outputs\compare\` metrics_table.csv, delong_tests.csv, oof_repeat_averaged.csv, figures\ (ROC, calibration,
  AUROC forest plot, phase-difference plot).
