$targets = @{
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors" = 6938041050
    "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"      = 2132625894
    "CyberRealistic_FINAL_FP16.safetensors"             = 2132651162
    "CyberRealisticPony_V18.0_F16.safetensors"          = 6938041288
}

$lastSizes = @{}
$lastTime = Get-Date

while ($true) {
    Clear-Host
    Write-Host "=== 下載進度與速率 ===" -ForegroundColor Cyan
    $now = Get-Date
    $elapsed = ($now - $lastTime).TotalSeconds
    if ($elapsed -le 0) { $elapsed = 1 }

    foreach ($name in $targets.Keys) {
        $target = $targets[$name]
        $partPath = "D:\AI-Image-Lab\ComfyUI\models\checkpoints\$name.part"
        $donePath = "D:\AI-Image-Lab\ComfyUI\models\checkpoints\$name"

        if (Test-Path $donePath) {
            Write-Host ("{0,-55} [完成] {1:N2} GB" -f $name, ($target/1GB)) -ForegroundColor Green
        }
        elseif (Test-Path $partPath) {
            $size = (Get-Item $partPath).Length
            $pct = [math]::Round(($size / $target) * 100, 1)

            $prev = $lastSizes[$name]
            $speed = 0
            if ($prev) {
                $speed = ($size - $prev) / $elapsed / 1MB
            }
            $lastSizes[$name] = $size

            $barLen = 30
            $filled = [math]::Round($barLen * $pct / 100)
            $bar = ("#" * $filled).PadRight($barLen, '-')

            Write-Host ("{0,-55}" -f $name)
            Write-Host ("  [{0}] {1,5:N1}%  {2,6:N2}/{3:N2} GB  {4,6:N2} MB/s" -f $bar, $pct, ($size/1GB), ($target/1GB), $speed)
        }
        else {
            Write-Host ("{0,-55} 等待中..." -f $name) -ForegroundColor DarkGray
        }
    }

    $lastTime = $now
    Start-Sleep -Seconds 2
}
