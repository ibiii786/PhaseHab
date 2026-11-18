Sources: `metadata/ISPY2-Analysis-mask-files-description.docx` and
`metadata/ISPY2-DCE-MRI-Data-Descriptions.pdf` (UCSF core lab, D. Newitt), plus our own check on 50 patients.

**How the masks were made (simple version).** The masks are the trial's "FTV analysis masks"
(FTV = functional tumour volume). They were not drawn by hand around the tumour. A program made them
with these steps:

1. **Box (manual).** A person drew a rectangular box around the tumour. They usually drew it on the
   **early post-contrast** image, or on the early-minus-pre subtraction image.
2. **Background cut (pre-contrast).** Dark voxels (fat, noise) were removed with a threshold on the
   **pre-contrast** image: 60% of the 95th-percentile brightness inside the box.
3. **Enhancement cut (early phase).** A voxel was kept only if it brightened enough on the
   **early** phase: percent enhancement PE = (early - pre) / pre >= a threshold, usually **70%**
   (for some patients the lab chose a different value by eye).
4. **Clean-up.** Voxels with too few kept neighbours were removed (connectivity filter).
5. **OMIT regions (manual, some patients).** Non-tumour enhancing areas (e.g. vessels) were cut out by hand.
6. For the FTV used in the trial, only voxels with a **positive SER** were counted. SER = (early - pre) / (late - pre),
   so this only asks that the **late** phase is brighter than pre-contrast.

The original files are "inverse", bit-coded masks (0 = tumour). The BreastDCEDL files we use are already
converted to simple binary masks (tumour > 0). For I-SPY1 (ACRIN 6657) the same method was used, with
thresholds set per imaging site.

**Do the masks depend on the early phase? Yes, strongly.**

- The documents say so directly: the main tumour/no-tumour decision is the early-phase enhancement threshold.
- Our own check (25 I-SPY1 + 25 I-SPY2 patients, random sample, seed 42) agrees.
  In the median patient, **100%** of mask voxels have early PE >= 70%. In a thin 3-voxel ring just
  outside the mask, only **~35%** do. **100%** of mask voxels are brighter on the late phase than on pre-contrast.
  A few I-SPY2 patients have a lower share (minimum 65%), which fits "threshold changed for some patients".
- The pre-contrast phase is also involved (background cut and the PE denominator). The late phase is
  involved only weakly (late > pre).

**What this means for PhaseHab (limitation to report).**

- The tumour region is *defined* by early-phase enhancement. Inside the mask, early-phase intensities are
  cut off at the low end (every voxel enhances >= ~70%). This can make early-phase features look
  different from pre and late features for a reason that has nothing to do with biology.
- This could favour or hurt the early phase. We cannot know which in advance, so we must say this
  clearly when comparing phases.
- Habitats use the 3 phase intensities, so every voxel already meets "early >= 1.7 x pre". Habitats will split
  the tumour *within* the enhancing region. They cannot find non-enhancing parts (e.g. necrosis), because
  those were removed by the mask.
- Shape features are the same for all three phases (one mask), so they cannot tell the phases apart.
  Keep them out of the phase comparison, or put them in a separate "shape" block.
- Possible sensitivity analysis: repeat the comparison with a slightly dilated mask (e.g. +1-2 mm). This
  loosens the link to the early-phase threshold.
- Note: the raw intensity scale differs a lot between cohorts (I-SPY1 max ~210, I-SPY2 max ~1500 in the sample).
  This supports the per-patient normalisation rule (all phases scaled with the same pre-contrast numbers).
