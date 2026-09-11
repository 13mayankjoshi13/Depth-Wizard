@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo   SIH26142 - Super-Resolution Mapping (GPU Setup)
echo =======================================================

echo [1/5] Checking Python installation (3.11 or 3.12 recommended)...
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=py -3.12
    goto :found_py
)

py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=py -3.11
    goto :found_py
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
    goto :found_py
)

echo ERROR: Python not found! Please install Python 3.11 or 3.12 from python.org and check "Add Python to PATH".
pause
exit /b 1

:found_py
echo Found Python: %PY_CMD%

echo.
echo [2/5] Creating virtual environment (.venv-gpu)...
if not exist .venv-gpu (
    %PY_CMD% -m venv .venv-gpu
) else (
    echo .venv-gpu already exists, updating packages...
)

echo.
echo [3/5] Upgrading pip and setuptools...
.venv-gpu\Scripts\python -m pip install --upgrade pip setuptools wheel

echo.
echo [4/5] Installing PyTorch with CUDA 12.1 (NVIDIA GPU acceleration)...
.venv-gpu\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [5/5] Installing dependencies (basicsr, realesrgan, rasterio, streamlit)...
.venv-gpu\Scripts\python -m pip install rasterio pyproj shapely opencv-python-headless scikit-image numpy Pillow scipy tqdm requests matplotlib plotly streamlit folium streamlit-folium streamlit-image-comparison pydeck pytest facexlib gfpgan

echo Installing basicsr and realesrgan...
.venv-gpu\Scripts\python -m pip install basicsr
if %ERRORLEVEL% NEQ 0 (
    echo Trying basicsr from GitHub...
    .venv-gpu\Scripts\python -m pip install "basicsr @ git+https://github.com/XPixelGroup/BasicSR.git"
)
.venv-gpu\Scripts\python -m pip install realesrgan

echo.
echo =======================================================
echo   Verifying GPU Detection:
echo =======================================================
.venv-gpu\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

echo.
echo =======================================================
echo   Setup Complete!
echo   To launch the dashboard, double-click: run_dashboard.bat
echo =======================================================
pause
