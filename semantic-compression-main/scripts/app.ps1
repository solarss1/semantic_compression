# Запуск desktop-застосунку (PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
    Write-Error "Спочатку виконайте: .\scripts\setup.ps1"
}

& "$Root\.venv\Scripts\python.exe" -m src.app
