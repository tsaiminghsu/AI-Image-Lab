@echo off
rem Double-click to start the AI Image Lab GUI at http://127.0.0.1:7861 (uses ComfyUI\.venv, the only environment with torch).
rem The browser opens by itself once the port answers (a hidden helper waits for it, at most about 5 minutes).
rem Close this window to stop the GUI. If a GUI already answers on the port, this just opens it in the browser.
cd /d "%~dp0training"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:7861/ | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [start_gui] A GUI is already running on port 7861 - opening it instead of starting a second one.
    start "" http://127.0.0.1:7861/
    timeout /t 3 >nul
    exit /b 0
)
if not exist "..\ComfyUI\.venv\Scripts\python.exe" (
    echo [start_gui] Cannot find ..\ComfyUI\.venv\Scripts\python.exe
    pause
    exit /b 1
)
start "" /min powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command "$url = 'http://127.0.0.1:7861/'; for ($i = 0; $i -lt 150; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $url | Out-Null; Start-Process $url; break } catch { Start-Sleep -Seconds 2 } }"
"..\ComfyUI\.venv\Scripts\python.exe" gui.py
echo.
echo [start_gui] The GUI has stopped.
pause
