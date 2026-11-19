@echo off
cd /d "%~dp0"
set "TF_FORCE_GPU_ALLOW_GROWTH=true"
py -3.10 -u "%~dp0train.py"
exit /b %errorlevel%
