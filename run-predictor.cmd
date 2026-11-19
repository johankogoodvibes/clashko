@echo off
cd /d "%~dp0"
set "TF_FORCE_GPU_ALLOW_GROWTH=true"
py -3.10 -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo The predictor requires Python 3.10 with Tcl/Tk. In the Python installer, choose Modify and enable tcl/tk and IDLE.
    pause
    exit /b 1
)
py -3.10 "%~dp0deck_predictor.py"
if errorlevel 1 pause
