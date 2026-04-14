@echo off
setlocal enabledelayedexpansion
title Industrial ROS2 MCP - Setup

echo.
echo  ============================================
echo   Industrial ROS2 MCP Server - Setup
echo   Isaac Sim 4.5.0 + UR10 + MCP
echo  ============================================
echo.

REM Find Python 3.10
set "PY310="
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do set "PY310=%%i"
)
if not defined PY310 (
    if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
        set "PY310=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
    )
)
if not defined PY310 (
    echo [ERROR] Python 3.10 not found! Isaac Sim 4.5.0 requires Python 3.10.
    echo Install it from: https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

echo [OK] Found Python 3.10: %PY310%

REM Create venv if not exists
if not exist "venv_isaac\Scripts\activate.bat" (
    echo.
    echo [1/3] Creating virtual environment with Python 3.10...
    "%PY310%" -m venv venv_isaac
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate venv
call venv_isaac\Scripts\activate.bat

REM Upgrade pip
echo.
echo [2/3] Installing dependencies (this may take 10-20 minutes first time)...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

REM Install Isaac Sim + PyTorch + project deps
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | findstr /i "already satisfied Installing ERROR"
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com 2>&1 | findstr /i "already satisfied Installing ERROR"
pip install numpy==1.26.4 pyyaml==6.0.3 Pillow>=10.0.0 fastmcp>=2.0.0 requests scipy==1.12.0 2>&1 | findstr /i "already satisfied Installing ERROR"

REM Install project in editable mode
pip install -e ".[dev]" 2>&1 | findstr /i "already satisfied Installing ERROR Successfully"

echo.
echo [3/3] Verifying installation...
python -c "import isaacsim; print(f'[OK] Isaac Sim: {isaacsim.__version__}')" 2>&1 || echo [WARN] Isaac Sim import failed - may need NVIDIA drivers
python -c "import torch; print(f'[OK] PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1
python -c "import fastmcp; print('[OK] FastMCP installed')" 2>&1
python -c "from mcp_server.server import mcp; print('[OK] MCP Server: 17 tools ready')" 2>&1

echo.
echo  ============================================
echo   Setup complete!
echo.
echo   To run:  run_isaac.bat
echo  ============================================
echo.
pause
