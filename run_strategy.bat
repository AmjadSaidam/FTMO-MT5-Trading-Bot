@echo off
:loop
"C:\FTMO_Live_Algo\.venv\Scripts\python.exe" "C:\FTMO_Live_Algo\live_trading\live_loop.py" >> "C:\FTMO_Live_Algo\logs\task_stdout.log" 2>&1
timeout /t 5
goto loop