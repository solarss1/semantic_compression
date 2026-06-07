# Перше налаштування (PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Write-Host "Завантаження U²-Net (потрібен інтернет)…"
python -m src.roi.u2net_setup
Write-Host "Готово. Запуск: .\scripts\app.ps1"
