@echo off
rem Double-click entry point for the click-based interactive segmentation app.
rem Switches to this folder, picks a Python interpreter, and runs main.py.

cd /d "%~dp0"

set "PY=..\SimpleClick\simpleclick_env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" main.py

if errorlevel 1 (
    echo.
    echo [error] The app did not start cleanly. See the message above.
    echo If a module is missing, run:  pip install -r requirements.txt
    pause
)
