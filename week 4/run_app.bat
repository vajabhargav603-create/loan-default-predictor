@echo off
echo ======================================================================
echo           LOANGUARD AI - WEEK 4 LOAN DEFAULT PREDICTION
echo ======================================================================
echo.
echo [1/2] Checking Python and starting Flask backend...
cd /d "%~dp0backend"

:: Start Flask server in background or foreground
echo.
echo [2/2] Opening Web Application in your default browser...
start http://127.0.0.1:5000

echo.
echo Running server on http://127.0.0.1:5000
echo Press Ctrl+C in this window to stop the server.
echo.
python app.py
pause
