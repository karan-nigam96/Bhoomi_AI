@echo off
REM BhoomiAI Flask Server Startup Script for Windows
REM This script starts the Flask development server and runs endpoint tests

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo BhoomiAI Flask Application Startup
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ and add it to your system PATH
    exit /b 1
)

echo [OK] Python interpreter found
echo.

REM Install requests if needed
echo [INFO] Checking for required packages...
python -m pip install requests flask scikit-learn pandas numpy -q

REM Run the test server script
echo [Starting Flask Server on http://localhost:5000]
echo.

python test_server_quick.py

pause
