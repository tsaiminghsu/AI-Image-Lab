# One-shot offline verification for this repo - the same checks CI runs:
#   ruff lint -> ruff format (tests only) -> pytest -> line-ending audit
#
# None of it needs a GPU, a running ComfyUI, or the model weights. Run it before every commit:
#   powershell -ExecutionPolicy Bypass -File check.ps1
#
#   -Live   also run tests marked "live" against a running ComfyUI (COMFYUI_URL)
#   -Fix    let ruff auto-fix and format first
param([switch]$Live, [switch]$Fix)

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv-dev\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "dev venv not found. Create it with:"
    Write-Host "    uv venv .venv-dev --python 3.11"
    Write-Host "    uv pip install --python .venv-dev\Scripts\python.exe -r requirements-dev.txt"
    Write-Host ""
    Write-Host "It is deliberately separate from ComfyUI\.venv so pytest/ruff can never be frozen"
    Write-Host "into comfyui-requirements.lock.txt and shipped in the RunPod worker image."
    exit 1
}

# The line-ending convention from CLAUDE.md, codified in .gitattributes and asserted here.
# README.md and training/poses/*.json are CRLF; everything else tracked as text is LF.
$crlfAllowed = '^(README\.md|training/poses/[^/]+\.json|training/reference_candidates/model_test/REPORT\.md)$'
$crlfRequired = '^(README\.md|training/poses/[^/]+\.json)$'

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    Write-Host "==> $Name"
    & $Body
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $Name"; exit 1 }
}

Push-Location $PSScriptRoot
try {
    if ($Fix) {
        Invoke-Step "ruff check --fix" { & $py -m ruff check --fix training tests worker }
        Invoke-Step "ruff format tests" { & $py -m ruff format tests }
    }
    Invoke-Step "ruff check" { & $py -m ruff check training tests worker }
    Invoke-Step "ruff format --check" { & $py -m ruff format --check tests }
    Invoke-Step "pytest" { & $py -m pytest }
    if ($Live) { Invoke-Step "pytest -m live" { & $py -m pytest -m live } }

    Write-Host "==> line endings"
    $eol = git ls-files --eol
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: git ls-files"; exit 1 }
    $bad = $eol | Where-Object { $_ -match '^i/crlf' } |
        ForEach-Object { ($_ -split '\s+')[3] } | Where-Object { $_ -notmatch $crlfAllowed }
    if ($bad) { Write-Host "unexpected CRLF in the index:"; $bad | ForEach-Object { Write-Host "  $_" }; exit 1 }
    $bad = $eol | Where-Object { $_ -match '^i/lf' } |
        ForEach-Object { ($_ -split '\s+')[3] } | Where-Object { $_ -match $crlfRequired }
    if ($bad) { Write-Host "expected CRLF but found LF:"; $bad | ForEach-Object { Write-Host "  $_" }; exit 1 }

    Write-Host ""
    Write-Host "all checks passed"
} finally {
    Pop-Location
}
