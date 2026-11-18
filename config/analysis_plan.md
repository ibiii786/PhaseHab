# PhaseHab analysis plan (fixed before any results)

Written 2026-11-18, after the cohort audit and before preprocessing, feature extraction or any model.
If we change anything later, we add a dated note at the bottom and say why. We do not edit the text above it.

## 1. Cohort
- 1,140 I-SPY patients (I-SPY1 158, I-SPY2 982) after removing 14 I-SPY1 patients with missing scans.
- **Exclude ISPY2-553012**: the mask covers only 1 slice (185 voxels), so habitats are not possible.
- **"Check in QC": ISPY2-726091, ISPY2-934459.** Their masks are coded 0/255, and they have 4× more voxels than the
  trial's `mask_count`. We keep them only if the QC overlay shows the mask sits on the tumour. The decision is
  recorded at the bottom of this file.

## 2. Splits
- Model development: **train (0) + validation (2) together** (about 1,010 patients).
- Repeated stratified 5-fold CV, **5 repeats**, seed 42. Strata = pCR × subtype (HER2+, TN, HR+/HER2−).
  The 3 patients with missing subtype are put in the HR+/HER2− stratum, for stratification only.
- **Test (1) is used once, at the very end**, with models frozen.

## 3. Resampling
- Real spacing from the metadata (`xy_spacing`, `slice_thick`), never from the NIfTI header.
- Resample to **1.0 × 1.0 × 1.0 mm**. Images: B-spline. Masks: nearest neighbour on `mask > 0`.

## 4. Normalisation (per patient; same numbers for all 3 phases)
- Foreground = pre-contrast voxels above the Otsu threshold, over the whole resampled crop.
- m, s = mean and std of the pre-contrast image over that foreground.
- Apply (x − m) / s to pre, early and late. Contrast uptake between phases is kept.
- We never normalise a phase on its own.

## 5. Habitats
- Voxel features = [pre, early, late] normalised intensities. ONE population K-means, k = 2, 3, 4.
- Clusters are ordered by mean early intensity, so habitat 1 = lowest enhancement.
- **Where it is fitted.** K-means is fitted **once on tumour voxels pooled from all train + validation patients**.
  The same habitat maps are used for the descriptive maps and for all CV folds. Why:
  (a) K-means never sees pCR labels, so it cannot leak the outcome. The test set is never used.
  (b) Refitting inside each of the 25 folds would mean 25 × 3 extra PyRadiomics runs, which is far too slow on this PC.
  **Check that the shortcut is safe:** we refit K-means on the training part of each fold of the first repeat,
  and compare its labels with the main fit (adjusted Rand index). If ARI < 0.90 in any fold, we switch to per-fold fitting.
- So that big tumours do not dominate, we take the same number of voxels from each patient
  (random 2,000 or all voxels if fewer, seed 42).
- **Minimum habitat size = 100 voxels (100 mm³).** If a habitat is smaller, its features are set to missing and
  imputed (median) inside the CV training folds. We report how many patients are affected for each k.

## 6. Feature blocks
- One PyRadiomics YAML (fixed bin width, IBSI-aligned) in `config\`, decided from training data before modelling.
- Per phase × per habitat: first-order + texture (GLCM, GLRLM, GLSZM, NGTDM, GLDM).
  Names: `<phase>_h<k>_<class>_<feature>`, e.g. `early_h2_glcm_Contrast`.
- **Shape block** (whole tumour + habitat volume fractions): only in the fused and clinical + imaging models,
  never in the per-phase comparison, because shape is the same for every phase.

## 7. Models
- **Clinical-only:** HR, HER2, age, cohort (I-SPY1 / I-SPY2).
  Sensitivity analysis only (I-SPY2 only): + treatment arm grouped as
  HER2-targeted (any trastuzumab, pertuzumab, T-DM1 or neratinib) / immunotherapy (pembrolizumab) /
  platinum (ABT 888 + carboplatin) / other (paclitaxel alone, AMG 386, ganitumab, ganetespib, MK-2206).
- **Imaging:** pre-only, early-only, late-only, fused (all 3 phases + shape), clinical + best phase.
- "Best phase" = highest CV AUROC on train + val. It is never chosen using the test set.
- Classifiers: logistic regression (L1/L2), SVM, random forest, XGBoost. Scaling, LASSO selection,
  imputation and tuning all happen inside the training folds. Class weights for imbalance.
- **Primary classifier for the phase tests (proposed): L2 logistic regression after LASSO selection.**
  It is fixed now, so we cannot pick whichever classifier gives the most significant result.
  The other classifiers are reported as secondary.

## 8. Statistics
- Main metric: out-of-fold AUROC. Each patient's OOF probability is averaged over the 5 repeats.
- DeLong test for paired AUROCs. Holm correction over **exactly 4 tests**:
  early vs pre, early vs late, late vs pre, fused vs best single phase.
- Also: AUPRC, Brier score, sensitivity / specificity / F1 at the Youden threshold (chosen in training folds),
  calibration curves. Bootstrap 95% CIs (2,000 resamples).
- Results for I-SPY1, I-SPY2 and pooled.
- **Minimum detectable AUROC difference, test set** (41 pCR+, 87 pCR−; Hanley–McNeil, paired, correlation 0.5,
  AUROC 0.70, power 80%): **≈ 0.15** at α = 0.05, **≈ 0.17** at the strictest Holm level (α = 0.0125).
  So the test set can only confirm large differences. The phase comparison relies mainly on the CV results,
  and the test set checks that performance holds up.

## 9. Known limitations
- Masks are the trial's FTV masks, defined by **early-phase enhancement ≥ ~70%** plus a pre-contrast background cut.
  The tumour region therefore depends on the early phase (see `docs\mask_origin.md`).
- NIfTI header spacing is wrong (all 1 mm). We use the metadata spacing instead.
- I-SPY1 is small (158 patients; 29 in test), so I-SPY1-only results will have wide CIs.
- Treatment arm exists only for I-SPY2.

## 10. Sensitivity analyses
1. Habitats from one phase only (K-means on early only), instead of all 3 phases.
2. Tumour mask dilated by 2 mm (loosens the link to the early-phase threshold).
3. I-SPY1 and I-SPY2 analysed separately.
4. Clinical model + grouped treatment arm (I-SPY2 only).

---

## Decisions log
(Dated entries added after this plan was fixed.)

### 2026-11-18 — QC after preprocessing (02_preprocess.py)
- **ISPY2-553012: excluded** (as planned). The QC overlay shows the mask covers only a small corner of a much larger
  enhancing tumour.
- **ISPY2-726091: kept.** The mask sits on the enhancing tumour.
- **ISPY2-934459: kept.** The mask covers the enhancing tumour, plus scattered small specks. Normal FTV masks have the same
  specks (60 random patients: median 40 separate pieces, largest piece = 93% of voxels; this patient: 87 pieces, 88%).
  The 4× difference from the trial's `mask_count` is probably a different counting or mask version; the mask itself is plausible.
- **ISPY1_1156: decision pending.** The metadata "early" phase (acq1) is a second pre-contrast scan (no enhancement:
  tumour mean 41.7 vs 41.9 on pre). The real early/late are acq2/acq3.
- **ISPY1_1187: decision pending.** The pre-contrast image has saturated voxels (>0.1% at 65,535; median 41), which
  breaks the Otsu normalisation (m = 43,935).

### 2026-11-18 — Decisions on the 2 pending patients (agreed with the user)
- **ISPY1_1156: kept, phases corrected** to pre = acq0, early = acq2, late = acq3 (`paths.PHASE_OVERRIDE`).
  After re-processing, the tumour enhances normally on early and washes out slightly on late (QC image checked).
- **ISPY1_1187: excluded** (`paths.EXCLUDE_ANALYSIS`). Its pre-contrast image is corrupted by saturated voxels.
- Analysis cohort is now **1,138 patients** (1,140 minus ISPY2-553012 and ISPY1_1187).
  `outputs\preprocessed\index.csv` has `use = False` for the 2 excluded patients and an `acq` column (pre/early/late).

### 2026-11-18 — Habitats (03_habitats.py)
- **K-means algorithm: standard (full) K-means instead of MiniBatchKMeans.** MiniBatch gave different clusters for
  different seeds on the same data (seed-vs-seed ARI 0.47–0.92), so it failed the stability check for a technical reason.
  Full K-means fits in < 30 s and in memory.
- **Stability check (plan section 5)**, refit on the training part of each fold of repeat 1, minimum ARI:
  k=2 0.973, **k=3 0.907 (pass)**, k=4 0.898 (just below 0.90). So for k=3 the single fit on train + val is kept.
  **If k=4 is used later, its K-means must be fitted inside each CV fold**, as the rule says.
- Habitats < 100 voxels (features = NaN, imputed in folds), k=3: h1 10, h2 42, **h3 380 patients (33%)**.

### 2026-11-18 — Model training details (05_train.py)
- Clinical-only model = HR, HER2, age, cohort (as in section 7). Treatment arm only in the I-SPY2 sensitivity model
  (`clinical_arm`, arm grouped as in section 7).
- Primary pipeline (lr), all inside the folds: median imputation → rank-to-normal scaling (QuantileTransformer;
  radiomics features are very skewed) → drop near-constant (one value ≥ 95%) → drop |r| > 0.95 (greedy, in column
  order) → LASSO (L1 logistic, C by inner 5-fold AUROC) → L2 logistic (C by inner 5-fold AUROC), class_weight balanced.
  The clinical model (4 inputs) skips the filters and LASSO.
- XGBoost (secondary): same imputation and filters, no LASSO, fixed settings (300 trees, depth 3, lr 0.05,
  subsample 0.8, colsample 0.5), balanced sample weights. Not tuned.
- "Best single phase" for `clinical_best` and the "fused vs best phase" test = **pre** (mean CV AUROC 0.566 vs
  late 0.566 vs early 0.553). Pre and late are practically tied (difference 0.0006); the rule picked pre.
