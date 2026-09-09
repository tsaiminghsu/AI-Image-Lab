# Stops the local ComfyUI server (127.0.0.1:8188) if one is running, freeing
# the ~2GB RAM/VRAM it holds even while idle. Complements gui.py's on-demand
# auto-start (_ensure_comfyui) - use this to shut ComfyUI back down when
# you're done generating instead of leaving it running until next reboot.

$conn = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue

if (-not $conn) {
    Write-Host "ComfyUI 目前沒有在跑（8188 埠沒有服務）。"
    exit 0
}

$procId = $conn.OwningProcess
$proc = Get-Process -Id $procId -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "找到監聽 8188 的紀錄，但對應的行程已經不存在了。"
    exit 0
}

Write-Host "正在關閉 ComfyUI（PID $procId, $([math]::Round($proc.WS/1MB,1)) MB）..."
Stop-Process -Id $procId -Force
Write-Host "已關閉。"
