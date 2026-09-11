#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-shot environment setup script for srm-project.
    Handles Python 3.12+ / 3.14 compatibility for basicsr/realesrgan.

.USAGE
    cd srm-project
    .\setup_env.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " SRM Project — Environment Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ─── Check Python version ─────────────────────────────────────────────────────
$pyver = python --version 2>&1
Write-Host "Python: $pyver"
$major, $minor = ($pyver -replace "Python ", "").Split(".")[0,1]
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
    Write-Error "Python 3.10+ required. Found: $pyver"
    exit 1
}

# ─── Create venv if missing ───────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment ..." -ForegroundColor Yellow
    python -m venv .venv
}
$pip = ".venv\Scripts\pip.exe"
$python = ".venv\Scripts\python.exe"

# ─── Upgrade pip / setuptools ─────────────────────────────────────────────────
Write-Host "`nUpgrading pip + setuptools ..." -ForegroundColor Yellow
& $pip install --upgrade pip setuptools wheel --quiet

# ─── PyTorch (CUDA build) ─────────────────────────────────────────────────────
Write-Host "`nInstalling PyTorch (CUDA 12.1) ..." -ForegroundColor Yellow
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA build failed — falling back to CPU-only PyTorch ..." -ForegroundColor DarkYellow
    & $pip install torch torchvision --quiet
}

# ─── Core scientific stack ────────────────────────────────────────────────────
Write-Host "`nInstalling core scientific packages ..." -ForegroundColor Yellow
& $pip install `
    rasterio `
    scikit-image `
    numpy `
    Pillow `
    scipy `
    opencv-python-headless `
    tqdm `
    requests `
    matplotlib `
    h5py `
    pyproj `
    shapely `
    --quiet

# ─── basicsr / realesrgan (Python 3.12+ compatibility note) ──────────────────
Write-Host "`nInstalling Real-ESRGAN + basicsr ..." -ForegroundColor Yellow
Write-Host "  Note: basicsr uses 'distutils' removed in Python 3.12+." -ForegroundColor DarkYellow
Write-Host "  Installing 'setuptools' first as a shim ..." -ForegroundColor DarkYellow
& $pip install setuptools --quiet   # Provides distutils shim for 3.12+
& $pip install facexlib gfpgan --quiet
& $pip install basicsr --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  basicsr install failed — trying from GitHub ..." -ForegroundColor DarkYellow
    & $pip install "basicsr @ git+https://github.com/XPixelGroup/BasicSR.git" --quiet
}
& $pip install realesrgan --quiet

# ─── Dashboard / UI ──────────────────────────────────────────────────────────
Write-Host "`nInstalling Streamlit + dashboard dependencies ..." -ForegroundColor Yellow
& $pip install `
    streamlit `
    folium `
    streamlit-folium `
    streamlit-image-comparison `
    pydeck `
    plotly `
    --quiet

# ─── Testing ─────────────────────────────────────────────────────────────────
Write-Host "`nInstalling test tools ..." -ForegroundColor Yellow
& $pip install pytest pytest-cov --quiet

# ─── Quick smoke test ─────────────────────────────────────────────────────────
Write-Host "`n--- Smoke test ---" -ForegroundColor Cyan
& $python -c @"
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

import rasterio
print(f'rasterio: {rasterio.__version__}')

import skimage
print(f'scikit-image: {skimage.__version__}')

import cv2
print(f'OpenCV: {cv2.__version__}')

import streamlit
print(f'streamlit: {streamlit.__version__}')

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print('basicsr: OK (RRDBNet importable)')
except Exception as e:
    print(f'basicsr: WARNING — {e}')

try:
    from realesrgan import RealESRGANer
    print('realesrgan: OK')
except Exception as e:
    print(f'realesrgan: WARNING — {e}')
"@

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Activate venv: .venv\Scripts\activate"
Write-Host "  2. Download tiles: python src\download_sentinel.py"
Write-Host "  3. Generate pairs: python src\pair_generation.py"
Write-Host "  4. Fine-tune: python src\train.py --epochs 50"
Write-Host "  5. Launch dashboard: streamlit run dashboard\app.py"
Write-Host "  6. Run tests: pytest tests\ -v"
