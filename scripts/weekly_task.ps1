# Wrapper invoked by Windows Task Scheduler. Runs the weekly benchmark
# pipeline and logs its output, since Task Scheduler runs are otherwise
# silent and hard to debug after the fact.

$RepoRoot = "C:\Users\GAMING\cv-logistics-mlops"
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("weekly_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

& uv run python scripts/weekly_pipeline.py *>&1 | Tee-Object -FilePath $LogFile

# keep only the 12 most recent logs
Get-ChildItem $LogDir -Filter "weekly_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 12 | Remove-Item -Force
