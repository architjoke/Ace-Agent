# PowerShell entrypoint. From repo root:
#   .\ace start
#   .\ace status
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
python agent.py @args
