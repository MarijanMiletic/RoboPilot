@echo off
echo ============================================
echo  Isaac Sim Bridge Server for MCP
echo  Listens on localhost:54321
echo ============================================
echo.

REM Activate your Isaac Sim venv if needed
REM call venv_isaac\Scripts\activate

python isaac_sim\bridge_server.py %*
pause
