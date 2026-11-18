"""All project paths in one place."""
from pathlib import Path

ROOT = Path(r"D:\Projects\FYP")
CODE = ROOT / "code"
OUT = CODE / "outputs"

# Data (read-only)
SPY1_DIR = ROOT / "BreastDCEDL_spy1" / "BreastDCEDL_spy1"
SPY1_DCE = SPY1_DIR / "spt1_dce"
SPY1_MASK = SPY1_DIR / "spy1_mask"
SPY2_DCE = ROOT / "ispy2_full" / "crop_min_dce"
SPY2_MASK = ROOT / "ispy2_full" / "crop_min_mask"
META = ROOT / "metadata"
META_MAIN = META / "BreastDCEDL_metadata_min_crop.csv"
META_SPY2 = META / "BreastDCEDL_spy2_tcia_metadata.csv"

# Patients excluded because their files are missing in the source data
EXCLUDE = {
    "ISPY1_1066", "ISPY1_1117", "ISPY1_1142", "ISPY1_1160", "ISPY1_1167",
    "ISPY1_1169", "ISPY1_1177", "ISPY1_1182", "ISPY1_1185", "ISPY1_1207",
    "ISPY1_1228", "ISPY1_1192", "ISPY1_1038", "ISPY1_1194",
}

# Excluded from the analysis by config/analysis_plan.md (still preprocessed, for QC)
EXCLUDE_ANALYSIS = {
    "ISPY2-553012",   # mask on 1 slice only, covers a small part of the tumour
    "ISPY1_1187",     # pre-contrast image has saturated voxels (65535): normalisation breaks
}
# "Check in QC" patients (0/255 masks, 4x the trial's mask_count) - both KEPT after QC
QC_CHECK = {"ISPY2-726091", "ISPY2-934459"}

# Corrections to the metadata phase choice (acquisition numbers), found in QC.
# ISPY1_1156: metadata "early" (acq1) is a second pre-contrast scan; real early/late = acq2/acq3.
PHASE_OVERRIDE = {"ISPY1_1156": {"early": 2, "late": 3}}

# Split codes in the metadata 'test' column
SPLIT_NAMES = {0: "train", 1: "test", 2: "val"}

PHASES = ["pre", "early", "late"]
PHASE_COLS = {"pre": "pre", "early": "post_early", "late": "post_late"}
SEED = 42


def phase_acq(row, phase: str) -> int:
    """Acquisition number for a phase: metadata value, unless corrected in PHASE_OVERRIDE."""
    fix = PHASE_OVERRIDE.get(row["pid"], {})
    return int(fix.get(phase, row[PHASE_COLS[phase]]))


def scan_path(pid: str, dataset: str, acq: int) -> Path:
    """Path of one DCE scan for a patient and acquisition number."""
    if dataset == "spy1":
        return SPY1_DCE / f"{pid}_spy1_vis1_acq{acq}.nii.gz"
    return SPY2_DCE / f"{pid}_spy2_vis1_dce_aqc_{acq}.nii.gz"


def mask_path(pid: str, dataset: str) -> Path:
    if dataset == "spy1":
        return SPY1_MASK / f"{pid}_spy1_vis1_mask.nii.gz"
    return SPY2_MASK / f"{pid}_spy2_vis1_mask.nii.gz"
