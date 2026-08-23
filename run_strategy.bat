@echo off
:loop
"C:\FTMO_Live_Algo\.venv\Scripts\python.exe" "C:\FTMO_Live_Algo\live_trading\live_loop.py"
timeout /t 5
goto loop