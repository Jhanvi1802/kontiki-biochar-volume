@echo off
title Kon-Tiki Biochar Volume - Web App
cd /d "%~dp0"
echo ================================================================
echo   Kon-Tiki Biochar Volume - Web App
echo ================================================================
echo.
echo Starting the app. When you see "Open http://127.0.0.1:5000",
echo open that address in your browser and upload a kiln photo.
echo (Try one from the sample_kiln_images folder.)
echo.
echo Close this window to stop the app.
echo.
python app.py
pause
