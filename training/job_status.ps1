# One-shot status of everything that runs in the background on this machine: the GPU, the services (Docker/Open WebUI,
# Ollama, the concept tool server, ComfyUI), the detached processes (downloads, tests), their latest log lines, disk and
# network. Read-only: it starts and stops nothing.
#
#   powershell -ExecutionPolicy Bypass -File training\job_status.ps1 [-Watch]     (-Watch refreshes every 5 s, Ctrl-C ends)

param([switch]$Watch)

$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "outputs\logs"

function Test-Json($url, $seconds = 3) {
    try { return Invoke-RestMethod $url -TimeoutSec $seconds } catch { return $null }
}

function Short($text, $n) {
    if ($null -eq $text) { return "" }
    $t = ($text -replace "\s+", " ").Trim()
    if ($t.Length -gt $n) { return $t.Substring(0, $n - 1) + "..." }
    return $t
}

function Show-Status {
    Write-Host ("=== 狀態 {0:yyyy-MM-dd HH:mm:ss} ===" -f (Get-Date)) -ForegroundColor Cyan

    Write-Host "`n[顯示卡]" -ForegroundColor Yellow
    $gpu = nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $p = $gpu -split ",\s*"
        Write-Host ("  {0}: {1}°C, 使用率 {2}%, 記憶體 {3}/{4} MiB, {5} W" -f $p[0], $p[1], $p[2], $p[3], $p[4], $p[5])
    } else { Write-Host "  讀不到 nvidia-smi" -ForegroundColor Red }

    Write-Host "`n[服務]" -ForegroundColor Yellow
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        $c = docker ps --filter name=open-webui --format "{{.Status}}"
        Write-Host ("  Open WebUI（localhost:3000）: {0}" -f $(if ($c) { $c } else { "容器沒有在跑" }))
    } else { Write-Host "  Docker: 沒有在跑" }
    $ollamaPort = if ($env:CONCEPT_OLLAMA_PORT) { $env:CONCEPT_OLLAMA_PORT } else { "21434" }
    $ps = Test-Json "http://127.0.0.1:$ollamaPort/api/ps"
    if ($null -ne $ps) {
        $loaded = @($ps.models | ForEach-Object { "{0}（{1:N1} GB）" -f ($_.name -replace "^hf\.co/.*:", ""), ($_.size / 1GB) })
        Write-Host ("  Ollama（{0}）: 在跑，載入中的模型：{1}" -f $ollamaPort, $(if ($loaded) { $loaded -join "、" } else { "無（顯示卡沒被它佔用）" }))
    } else { Write-Host ("  Ollama（{0}）: 沒有回應" -f $ollamaPort) }
    $jobs = Test-Json "http://127.0.0.1:7863/jobs?limit=20"
    if ($null -ne $jobs) {
        $active = @($jobs.jobs | Where-Object { $_.status -in @("queued", "running", "waiting_for_user") })
        Write-Host ("  工具伺服器（7863）: 在跑，未完成的工作 {0} 個" -f $active.Count)
        foreach ($j in $active) { Write-Host ("    {0} 故事{1} {2} {3}" -f $j.job_id, $j.story, $j.status, (Short $j.message 60)) }
    } else { Write-Host "  工具伺服器（7863）: 沒有在跑" }
    foreach ($svc in @(@{ n = "ComfyUI"; u = "http://127.0.0.1:9188/system_stats" }, @{ n = "一站式頁面"; u = "http://127.0.0.1:7864/" })) {
        $up = $null -ne (Test-Json $svc.u 2)
        Write-Host ("  {0}: {1}" -f $svc.n, $(if ($up) { "在跑" } else { "沒有在跑" }))
    }

    Write-Host "`n[背景程序]" -ForegroundColor Yellow
    $pattern = "fetch_|turbo_test|qwen21_|concept_api|studio_api|main\.py|ollama|pip install|check\.ps1|comfyui|benchmark|bench\.py|submit\.py|worker\.py"
    $mine = $PID
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @("python.exe", "uv.exe", "ollama.exe", "ffmpeg.exe", "curl.exe", "docker.exe") -and
        $_.CommandLine -match $pattern -and $_.ProcessId -ne $mine -and $_.CommandLine -notmatch "mcpforun"
    }
    if ($procs) {
        foreach ($x in ($procs | Sort-Object CreationDate)) {
            $age = (Get-Date) - $x.CreationDate
            # name the job by its script, not by the long path of the interpreter that runs it
            $what = if ($x.CommandLine -match '([^\\/\s"]+\.(?:py|ps1))') { $Matches[1] } else { Short ($x.CommandLine -replace '^"?[^ ]*\\', '') 60 }
            Write-Host ("  {0,6} {1,-11} 已跑 {2,-8} {3}" -f $x.ProcessId, $x.Name, ("{0:N0}h{1:00}m" -f [math]::Floor($age.TotalHours), $age.Minutes), $what)
        }
    } else { Write-Host "  沒有相關的背景程序" }

    Write-Host "`n[最近的日誌（outputs\logs）]" -ForegroundColor Yellow
    if (Test-Path $logDir) {
        $recent = Get-ChildItem $logDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 0 -and $_.LastWriteTime -gt (Get-Date).AddHours(-24) -and $_.Extension -in ".log", ".err" } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 6
        foreach ($f in $recent) {
            $last = Get-Content $f.FullName -Tail 1 -ErrorAction SilentlyContinue
            $mins = [math]::Round(((Get-Date) - $f.LastWriteTime).TotalMinutes)
            Write-Host ("  {0} （{1} 分鐘前）: {2}" -f $f.Name, $mins, (Short $last 100))
        }
        if (-not $recent) { Write-Host "  24 小時內沒有日誌更新" }
    }

    Write-Host "`n[磁碟與網路]" -ForegroundColor Yellow
    $d = Get-PSDrive D -ErrorAction SilentlyContinue
    if ($d) { Write-Host ("  D: 剩 {0:N0} GB" -f ($d.Free / 1GB)) }
    $net = (Get-Counter '\Network Interface(*)\Bytes Received/sec' -ErrorAction SilentlyContinue).CounterSamples | Sort-Object CookedValue -Descending | Select-Object -First 1
    if ($net) { Write-Host ("  目前下載速度 {0:N1} MB/s" -f ($net.CookedValue / 1MB)) }
}

if ($Watch) {
    while ($true) { Clear-Host; Show-Status; Start-Sleep 5 }
} else {
    Show-Status
}
