# One-click start for the short-drama concept flow in Open WebUI:
#   Docker Desktop -> open-webui container -> concept_api tool server (127.0.0.1:7863) -> browser.
# It never starts ComfyUI or touches the GPU (the tool server starts ComfyUI itself, only after a confirmed
# generate_candidates). The Open WebUI registration of the tool server lives in the container's volume, so it
# only has to be done once; this script only brings the pieces up.
#
#   powershell -ExecutionPolicy Bypass -File training\start_concept_webui.ps1 [-NoBrowser]

param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$port = if ($env:CONCEPT_API_PORT) { $env:CONCEPT_API_PORT } else { "7863" }
$apiUrl = "http://127.0.0.1:$port"
$webUrl = "http://localhost:3000"
$container = "open-webui"
$python = Join-Path $repo "ComfyUI\.venv\Scripts\python.exe"
$dockerExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$logDir = Join-Path $repo "outputs\logs"

# Port 8188 is reserved by Windows on this machine, so the tool server's own ComfyUI uses 9188.
if (-not $env:COMFYUI_URL) { $env:COMFYUI_URL = "http://127.0.0.1:9188" }

function Test-Url($url, $seconds = 5) {
    try { return (Invoke-WebRequest -UseBasicParsing $url -TimeoutSec $seconds).StatusCode -eq 200 } catch { return $false }
}

function Wait-Until($label, [scriptblock]$check, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        if (& $check) { return $true }
        Start-Sleep 3
    }
    Write-Host "  等了 $seconds 秒仍然沒好：$label" -ForegroundColor Red
    return $false
}

function Stop-Here($message) {
    Write-Host $message -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $python)) { Stop-Here "找不到 $python" }

# 1. Docker Desktop
Write-Host "[1/4] Docker Desktop"
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path $dockerExe)) { Stop-Here "Docker 沒有在跑，也找不到 Docker Desktop，請自己開啟後再執行。" }
    Write-Host "  Docker 沒開，正在啟動（最多等 3 分鐘）..."
    Start-Process $dockerExe
    $ready = Wait-Until "Docker 就緒" { docker info *> $null; $LASTEXITCODE -eq 0 } 180
    if (-not $ready) { Stop-Here "Docker Desktop 沒有在時間內就緒。" }
}
Write-Host "  Docker 已就緒"

# 2. Open WebUI container (created earlier; this never creates or deletes one)
Write-Host "[2/4] Open WebUI 容器"
docker inspect $container *> $null
if ($LASTEXITCODE -ne 0) { Stop-Here "找不到名為 $container 的容器，請先照 OPEN_WEBUI.md 建立。" }
if ((docker inspect $container --format "{{.State.Running}}") -ne "true") {
    docker start $container | Out-Null
}
$up = Wait-Until "Open WebUI 回應" { Test-Url $webUrl } 150
if (-not $up) { Stop-Here "Open WebUI 沒有回應 $webUrl" }
Write-Host "  Open WebUI 已就緒 $webUrl"

# 3. Tool server, detached so it survives this window and the Claude session
Write-Host "[3/4] 工具伺服器（concept_api）"
if (Test-Url "$apiUrl/openapi.json") {
    Write-Host "  已經在跑，沿用"
} else {
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $log = Join-Path $logDir "concept_api.log"
    Start-Process -FilePath $python -ArgumentList "training\concept_api.py" -WorkingDirectory $repo `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $ok = Wait-Until "工具伺服器回應" { Test-Url "$apiUrl/openapi.json" } 60
    if (-not $ok) {
        if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 15 }
        Stop-Here "工具伺服器沒有起來，記錄在 $log"
    }
    Write-Host "  已啟動 $apiUrl（記錄 $log）"
}

# 4. Can the container reach the tool server, and is the chat model's backend (Ollama) up?
Write-Host "[4/4] 連線檢查"
$probe = "import urllib.request;print(urllib.request.urlopen('http://host.docker.internal:$port/openapi.json',timeout=10).status)"
$reach = (docker exec $container python -c $probe 2>$null | Select-Object -Last 1)
if ($reach -eq "200") { Write-Host "  容器連得到工具伺服器" } else { Write-Host "  容器連不到工具伺服器（host.docker.internal:$port）" -ForegroundColor Yellow }
$ollamaUrl = "http://127.0.0.1:11434/api/tags"
if (-not (Test-Url $ollamaUrl)) {
    # The tray app can be running without its server. `ollama serve` only opens the API; it loads no model, so the GPU stays idle.
    $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if ($ollama) {
        Write-Host "  Ollama 服務沒開，正在啟動（不會載入模型）..."
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
        Wait-Until "Ollama 回應" { Test-Url $ollamaUrl } 30 | Out-Null
    }
}
if (Test-Url $ollamaUrl) { Write-Host "  Ollama 在跑" } else { Write-Host "  Ollama 沒有回應：聊天模型會用不了，請先開 Ollama" -ForegroundColor Yellow }

Write-Host ""
Write-Host "完成。顯示卡沒有被動用；要生圖時由聊天裡的工具啟動 ComfyUI。" -ForegroundColor Green
Write-Host "提醒：生圖期間盡量不要聊天（聊天模型和生圖共用同一張 8 GB 顯示卡）。"
if (-not $NoBrowser) { Start-Process $webUrl }
