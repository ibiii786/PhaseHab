@echo off
REM PhaseHab - overnight run for k = 2 and k = 4 habitats.
REM   1-2. radiomics features for k = 2 and k = 4 (the slow part, about 3 hours)
REM   3-4. cross-validated training for k = 2 and k = 4 (about 10-15 min each)
REM   5.   choose k on cross-validation results -> outputs\models\k_selection.md
REM Runs only on this PC: no internet and no Claude needed. The test set is never used.
REM Double-click to start. Close the window to stop. Double-click again to continue:
REM finished steps and finished patients are skipped.
title PhaseHab - overnight k=2 / k=4
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set PYTHONUNBUFFERED=1

echo ============================================================
echo  PhaseHab overnight run: k = 2 and k = 4  (about 3.5 hours)
echo  Main log:  %~dp0outputs\overnight_k2_k4.log
echo  Result:    %~dp0outputs\models\k_selection.md
echo.
echo  - Keep the PC plugged in and switched on. Do not close the laptop lid.
echo    (While this runs, the program asks Windows not to go to sleep.)
echo  - To stop: close this window. To continue: double-click again.
echo ============================================================
echo.

".venv\Scripts\python.exe" src\run_overnight.py
set RC=%ERRORLEVEL%

echo.
if %RC%==0 (
  echo FINISHED OK. Tell Claude that the overnight k2/k4 run is done.
) else (
  echo STOPPED WITH AN ERROR ^(code %RC%^). Look at outputs\overnight_k2_k4.log, or double-click again to retry.
)
echo.
pause
