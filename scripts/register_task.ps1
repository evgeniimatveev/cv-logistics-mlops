# Registers (or re-registers) the weekly benchmark automation in Windows
# Task Scheduler. Run once manually:
#   powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
# Change $DayOfWeek / $Time below to pick a different schedule.

$TaskName = "CVLogisticsWeeklyBenchmark"
$ScriptPath = "C:\Users\GAMING\cv-logistics-mlops\scripts\weekly_task.ps1"
$DayOfWeek = "Sunday"
$Time = "09:00AM"

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayOfWeek -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Weekly auto-benchmark for cv-logistics-mlops: trains the next comparison config, updates BENCHMARKS.md, pushes to GitHub." `
    -User $env:USERNAME

Write-Output "Registered '$TaskName': every $DayOfWeek at $Time (runs at next login if the PC was off)."
