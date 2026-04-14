@echo off
setlocal
title Isaac Sim Bridge Server - UR10 MCP

echo.
echo  ============================================
echo   Isaac Sim Bridge Server
echo   UR10 Robot + MCP on localhost:54321
echo  ============================================
echo.

REM Activate venv
if not exist "venv_isaac\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Run setup.bat first.
    pause
    exit /b 1
)

call venv_isaac\Scripts\activate.bat

REM Windows fix for torch
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting Isaac Sim with UR10 robot...
echo Robot will be controllable via Claude Desktop.
echo.
echo   Close this window to stop the simulation.
echo.

python isaac_sim\bridge_server.py

pause
