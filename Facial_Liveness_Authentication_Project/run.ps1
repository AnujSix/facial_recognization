$ErrorActionPreference = "Stop"
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "Virtual environment (.venv) not found. Run uv or python setup first."
    exit 1
}
if ($args.Count -gt 0 -and $args[0] -eq "demo") {
    $remaining = $args | Select-Object -Skip 1
    & $venvPy (Join-Path $PSScriptRoot "demo_pipeline.py") @remaining
} else {
    & $venvPy (Join-Path $PSScriptRoot "app.py") @args
}
