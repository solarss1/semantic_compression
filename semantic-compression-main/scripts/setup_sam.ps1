# Ultralytics SAM (PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    & "$PSScriptRoot\setup.ps1"
}

& .\.venv\Scripts\Activate.ps1
pip install -r requirements-sam.txt
pip install -e ".[sam]"
python -m src.roi.ultralytics_sam_setup --download
Write-Host "Готово. У GUI оберіть «Ultralytics SAM»."
