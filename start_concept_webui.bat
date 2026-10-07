@echo off
rem Double-click to bring up Docker, Open WebUI and the concept tool server, then open the browser.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "training\start_concept_webui.ps1"
echo.
pause
