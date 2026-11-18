r"""
run_overnight.py - Run the k = 2 and k = 4 analysis unattended (used by run_k2_k4_overnight.bat).

Steps, in order (each one is skipped if it is already finished, so you can restart after a crash or reboot):
  1. 04_features.py --k 2   radiomics features, k = 2  (resumes by itself: finished patients are skipped)
  2. 04_features.py --k 4   radiomics features, k = 4
  3. 05_train.py    --k 2   cross-validation + refit    (skipped if outputs/models/k2/DONE.txt exists)
  4. 05_train.py    --k 4
  5. choose_k.py            compares k = 2, 3, 4 on CV results -> outputs/models/k_selection.md

Everything runs on this PC only: no internet, no Claude. The test set is never used.
While it runs, Windows is asked not to go to sleep (the request ends when this program ends).
Log: outputs/overnight_k2_k4.log (plus the normal log of each step).

Run from code\ :  .venv\Scripts\python.exe src\run_overnight.py [--dry-run]
"""
import argparse
import ctypes
import datetime as dt
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import cv  # noqa: E402
import paths as P  # noqa: E402

PY = sys.executable
LOG = P.OUT / "overnight_k2_k4.log"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake(on):
    """Ask Windows not to sleep while this program runs (no settings are changed)."""
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:  # noqa: BLE001  (not on Windows)
        pass


def features_complete(k):
    """True if features/k{k}/early.csv has every analysis patient."""
    f = P.OUT / "features" / f"k{k}" / "early.csv"
    if not f.exists():
        return False
    import pandas as pd
    n = len(pd.read_csv(f, usecols=["pid"]))
    return n == len(cv.load_index())


def run_step(name, args, step_log=None):
    """Run one script. Output goes to this window; if step_log is given it is also saved there."""
    log(f"START  {name}:  {' '.join(args)}")
    t0 = time.time()
    if step_log is None:
        rc = subprocess.call([PY] + args)
    else:
        step_log.parent.mkdir(parents=True, exist_ok=True)
        with open(step_log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([PY] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
            rc = proc.wait()
    mins = (time.time() - t0) / 60
    if rc != 0:
        log(f"FAILED {name} (exit code {rc}) after {mins:.1f} min")
        return False
    log(f"END    {name} in {mins:.1f} min")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only show what would run")
    args = ap.parse_args()
    os.chdir(P.CODE)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    steps = []
    for k in [2, 4]:
        steps.append((f"features k={k}", ["src/04_features.py", "--k", str(k), "--jobs", "4"],
                      lambda k=k: features_complete(k), None))
    for k in [2, 4]:
        steps.append((f"training k={k}", ["src/05_train.py", "--k", str(k), "--jobs", "4"],
                      lambda k=k: (P.OUT / "models" / f"k{k}" / "DONE.txt").exists(),
                      P.OUT / "models" / f"k{k}" / "run_05_train.log"))
    steps.append(("choose k", ["src/choose_k.py"], lambda: False, P.OUT / "models" / "choose_k.log"))

    if args.dry_run:
        for name, a, done, _ in steps:
            print(f"{name:16s} {'already done -> skip' if done() else 'will run'}   ({' '.join(a)})")
        return 0

    log("=" * 70)
    log("OVERNIGHT RUN k = 2 and k = 4 started")
    keep_awake(True)
    try:
        for name, a, done, step_log in steps:
            if done():
                log(f"SKIP   {name} (already finished)")
                continue
            if not run_step(name, a, step_log):
                log("STOPPED. Fix the problem (or just double-click the .bat again to retry).")
                return 1
            if name.startswith("features") and not done():
                log(f"FAILED {name}: the feature tables do not have every patient yet. Double-click again.")
                return 1
        log("ALL STEPS FINISHED OK. Summary: outputs\\models\\k_selection.md")
        return 0
    finally:
        keep_awake(False)


if __name__ == "__main__":
    sys.exit(main())
