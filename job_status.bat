@echo off
rem Double-click: shows the GPU, services, background processes and latest logs once, then waits.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "training\job_status.ps1"
echo.
pause
