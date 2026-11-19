@echo off
setlocal
cd /d "%~dp0"
if /i "%~1"=="--worker" goto worker

if not exist "data\starting_players.txt" (
    echo Missing data\starting_players.txt.
    exit /b 1
)
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 is unavailable. Check that py -3.10 works.
    exit /b 1
)

set "COLLECTION_STATUS_DIR=%CD%\.tools\collection-%RANDOM%-%RANDOM%"
mkdir "%COLLECTION_STATUS_DIR%" || exit /b 1
set "STARTED=0"
set "LAUNCH_FAILED=0"
for /f "usebackq tokens=1-3" %%A in ("data\starting_players.txt") do call :launch "%%A" "%%B" "%%C"
echo Opened %STARTED% collector window(s). Each window reports its final result.
echo Status files: %COLLECTION_STATUS_DIR%
if "%STARTED%"=="0" exit /b 1
exit /b %LAUNCH_FAILED%

:launch
if "%~2"=="" goto invalid_row
if "%~3"=="" goto invalid_row
if defined COLLECTING_%~3 (
    echo Skipped %~1: MMR %~3 already has a worker in this launch.
    set "LAUNCH_FAILED=1"
    exit /b
)
start "Collect %~1 at %~3 MMR" "%ComSpec%" /d /k call "%~f0" --worker "%~1" "%~2" "%~3"
if errorlevel 1 (
    echo Failed to open a window for %~1.
    set "LAUNCH_FAILED=1"
    exit /b
)
set "COLLECTING_%~3=1"
set /a STARTED+=1 >nul
exit /b

:invalid_row
echo Skipped %~1: expected name, player tag, and MMR.
set "LAUNCH_FAILED=1"
exit /b

:worker
set "STATUS_FILE=%COLLECTION_STATUS_DIR%\%~4.txt"
echo [RUNNING] %~2 %~3 at %~4 MMR
>"%STATUS_FILE%" echo [RUNNING] %~2 %~3 at %~4 MMR
py -3.10 -u "data\collect_mmr_bubble.py" --tag "%~3" --mmr "%~4" --require-target
set "RESULT=%ERRORLEVEL%"
set "MESSAGE=[FAILED] %~2 at %~4 MMR. Exit code %RESULT%."
if "%RESULT%"=="0" set "MESSAGE=[OK] %~2 at %~4 MMR reached the match target."
if "%RESULT%"=="3" set "MESSAGE=[PARTIAL] %~2 at %~4 MMR stopped below the match target. See the collector summary above."
echo %MESSAGE%
>"%STATUS_FILE%" echo %MESSAGE%
exit /b %RESULT%
