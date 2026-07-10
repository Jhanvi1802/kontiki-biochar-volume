@echo off
title Kon-Tiki Biochar Volume - Video Dashboard
cd /d "%~dp0"
echo ================================================================
echo   Kon-Tiki Biochar Volume - Video Dashboard  (local)
echo ================================================================
echo.
echo Starting... when you see "Open http://127.0.0.1:5001",
echo open that address in your browser.
echo   - Upload dense.ply  -> biochar volume (runs locally on your PC)
echo   - Upload a video    -> local frame extraction + next-step hand-off
echo.
echo Close this window to stop the dashboard.
echo.
"C:\Users\Admin\kontiki_venv\Scripts\python.exe" app_video.py
pause
