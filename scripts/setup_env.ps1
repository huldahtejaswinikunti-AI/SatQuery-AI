# ==============================================================================
# SatQuery AI -- Complete Environment Setup Script (Windows PowerShell)
# SIH 2026 PS 26167 (ISRO/SAC)
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Setting up SatQuery AI Environment (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "[1/5] Creating Python virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/5] Virtual environment .venv already exists." -ForegroundColor Green
}

# Activate venv if script exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
}

# 2. Upgrade pip and install core dependencies
Write-Host "[3/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest fpdf2 rasterio streamlit

# 3. Install satquery package in editable mode
Write-Host "[4/5] Installing satquery in editable mode (-e .)..." -ForegroundColor Yellow
pip install -e .

# 4. Generate demo samples and metadata
Write-Host "[5/5] Generating curated demonstration samples..." -ForegroundColor Yellow
python data\scripts\download_demo_samples.py

# 5. Verification
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
Write-Host "Running verification smoke tests..." -ForegroundColor Yellow

$smoke = python scripts\smoke_test.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " SatQuery AI environment setup SUCCESSFUL!" -ForegroundColor Green
    Write-Host " Launch UI with:  streamlit run app\main.py" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host " Verification failed. Please check the logs above." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    exit 1
}
