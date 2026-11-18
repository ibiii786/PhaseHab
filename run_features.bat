@echo off
REM PhaseHab - full radiomics feature extraction (k = 3).
REM Double-click to start. Close the window to stop. Double-click again to continue:
REM patients that are already finished are skipped.
title PhaseHab - radiomics features (k=3)
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set PYTHONUNBUFFERED=1

echo ============================================================
echo  PhaseHab feature extraction (k = 3, 4 workers)
echo  Log file: %~dp0outputs\features\k3\run_features.log
echo  Results:  %~dp0outputs\features\k3\  (pre/early/late/shape.csv)
echo.
echo  - Keep the PC switched on and stop it from going to sleep.
echo  - To stop: close this window. To continue: double-click again.
echo ============================================================
echo.

".venv\Scripts\python.exe" src\04_features.py --k 3 --jobs 4
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
  echo FINISHED OK. Tell Claude that the feature run is done.
) else (
  echo STOPPED WITH AN ERROR ^(code %RC%^). Look at the log file above, or double-click again to retry.
)
echo.
pause
