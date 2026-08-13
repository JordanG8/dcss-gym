@echo off
title DCSS Dashboard
cd /d "%~dp0"

REM Start the control panel only if it isn't already listening on 8099.
powershell -NoProfile -Command ^
  "try { $null = Invoke-WebRequest -Uri 'http://localhost:8099/' -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }"

if errorlevel 1 (
    echo Starting control panel...
    REM Output MUST be redirected to files. Without it the server inherits this
    REM console's handles and is killed the moment this window closes.
    powershell -NoProfile -Command ^
      "Start-Process -FilePath 'python' -ArgumentList 'project.py','--serve' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput '%~dp0panel.out.log' -RedirectStandardError '%~dp0panel.err.log'"
    powershell -NoProfile -Command "Start-Sleep -Seconds 5"
) else (
    echo Control panel already running.
)

start "" http://localhost:8099
exit
