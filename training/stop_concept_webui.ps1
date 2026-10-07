# Stops the concept tool server (concept_api). Leaves Docker and Open WebUI alone.
# Refuses while a job is queued/running/waiting for you, because a restart marks it `interrupted` and
# never reruns it; pass -Force to stop anyway.
#
#   powershell -ExecutionPolicy Bypass -File training\stop_concept_webui.ps1 [-Force]

param([switch]$Force)

$port = if ($env:CONCEPT_API_PORT) { $env:CONCEPT_API_PORT } else { "7863" }
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $conn) {
    Write-Host "工具伺服器目前沒有在跑（$port 埠沒有服務）。"
    exit 0
}

if (-not $Force) {
    try {
        $jobs = (Invoke-RestMethod "http://127.0.0.1:$port/jobs?limit=50" -TimeoutSec 5).jobs
        $busy = @($jobs | Where-Object { $_.status -in @("queued", "running", "waiting_for_user") })
        if ($busy.Count -gt 0) {
            Write-Host "還有 $($busy.Count) 個工作沒做完，先不關：" -ForegroundColor Yellow
            $busy | ForEach-Object { Write-Host "  $($_.job_id)  $($_.kind)  $($_.story)  $($_.status)  $($_.message)" }
            Write-Host "要硬關請加 -Force（沒做完的工作會標成 interrupted，不會自動重跑）。"
            exit 1
        }
    } catch {
        Write-Host "查不到工作狀態（$($_.Exception.Message)），不加 -Force 就不關。" -ForegroundColor Yellow
        exit 1
    }
}

$procId = $conn.OwningProcess
Write-Host "正在關閉工具伺服器（PID $procId）..."
Stop-Process -Id $procId -Force
Write-Host "已關閉。"
