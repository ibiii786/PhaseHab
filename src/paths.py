from pathlib import Path

ROOT = Path(r"D:\Projects\FYP")
CODE = ROOT / "code"
OUT = CODE / "outputs"

SPY1_DIR = ROOT / "BreastDCEDL_spy1" / "BreastDCEDL_spy1"
SPY1_DCE = SPY1_DIR / "spt1_dce"
SPY1_MASK = SPY1_DIR / "spy1_mask"
SPY2_DCE = ROOT / "ispy2_full" / "crop_min_dce"
SPY2_MASK = ROOT / "ispy2_full" / "crop_min_mask"
META = ROOT / "metadata"
META_MAIN = META / "BreastDCEDL_metadata_min_crop.csv"
META_SPY2 = META / "BreastDCEDL_spy2_tcia_metadata.csv"

EXCLUDE = {
    "ISPY1_1066", "ISPY1_1117", "ISPY1_1142", "ISPY1_1160", "ISPY1_1167",
    "ISPY1_1169", "ISPY1_1177", "ISPY1_1182", "ISPY1_1185", "ISPY1_1207",
    "ISPY1_1228", "ISPY1_1192", "ISPY1_1038", "ISPY1_1194",
}

EXCLUDE_ANALYSIS = {
    "ISPY2-553012",
    "ISPY1_1187",
}
QC_CHECK = {"ISPY2-726091", "ISPY2-934459"}

PHASE_OVERRIDE = {"ISPY1_1156": {"early": 2, "late": 3}}

SPLIT_NAMES = {0: "train", 1: "test", 2: "val"}

PHASES = ["pre", "early", "late"]
PHASE_COLS = {"pre": "pre", "early": "post_early", "late": "post_late"}
SEED = 42


def phase_acq(row, phase: str) -> int:
    fix = PHASE_OVERRIDE.get(row["pid"], {})
    return int(fix.get(phase, row[PHASE_COLS[phase]]))


def scan_path(pid: str, dataset: str, acq: int) -> Path:
    if dataset == "spy1":
        return SPY1_DCE / f"{pid}_spy1_vis1_acq{acq}.nii.gz"
    return SPY2_DCE / f"{pid}_spy2_vis1_dce_aqc_{acq}.nii.gz"


def mask_path(pid: str, dataset: str) -> Path:
    if dataset == "spy1":
        return SPY1_MASK / f"{pid}_spy1_vis1_mask.nii.gz"
    return SPY2_MASK / f"{pid}_spy2_vis1_mask.nii.gz"
