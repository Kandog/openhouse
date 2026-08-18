@echo off
cd /d "%~dp0"
echo Starting Openhouse AI Assistant...
echo.
if exist ".venv313\Scripts\python.exe" (
    echo Using Python 3.13 virtual environment
    .venv313\Scripts\python.exe main.py
) else if exist ".venv\Scripts\python.exe" (
    echo Using Python virtual environment
    .venv\Scripts\python.exe main.py
) else (
    echo ERROR: No virtual environment found!
    echo Run: .venv313\Scripts\python.exe main.py
    pause
    exit /b 1
)
