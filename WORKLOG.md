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
