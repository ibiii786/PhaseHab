@echo off
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
