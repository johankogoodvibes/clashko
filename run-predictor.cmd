@echo off
cd /d "%~dp0"
set "TF_FORCE_GPU_ALLOW_GROWTH=true"
set "PATH=%~dp0.venv\Library\bin;%~dp0.venv;%~dp0.venv\Scripts;%PATH%"
"%~dp0.venv\python.exe" "%~dp0deck_predictor.py"
if errorlevel 1 pause
