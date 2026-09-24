@echo off
set "ROOT=%~dp0"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
:loop
"%ROOT%.venv\Scripts\python.exe" "%ROOT%live_trading\live_loop.py" >> "%ROOT%logs\task_stdout.log" 2>&1
timeout /t 5
goto loop