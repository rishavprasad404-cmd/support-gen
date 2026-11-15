@echo off
REM One-click start for Windows. Double-click this file after extracting the zip.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed or not on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist .env copy .env.example .env >nul
echo.
echo Starting SupportGen at http://localhost:5000  (press Ctrl+C to stop)
start "" http://localhost:5000
python app.py
pause
