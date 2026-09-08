#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI -- Complete Environment Setup Script (Linux / macOS)
# SIH 2026 PS 26167 (ISRO/SAC)
# ==============================================================================

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} Setting up SatQuery AI Environment (Linux / macOS)${NC}"
echo -e "${CYAN}============================================================${NC}"

# 1. Virtual Environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[1/5] Creating Python virtual environment in .venv...${NC}"
    python3 -m venv .venv
else
    echo -e "${GREEN}[1/5] Virtual environment .venv already exists.${NC}"
fi

# Activate venv
echo -e "${YELLOW}[2/5] Activating virtual environment...${NC}"
source .venv/bin/activate

# 2. Upgrade pip and install core dependencies
echo -e "${YELLOW}[3/5] Installing dependencies from requirements.txt...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest fpdf2 rasterio streamlit

# 3. Install satquery package in editable mode
echo -e "${YELLOW}[4/5] Installing satquery in editable mode (-e .)...${NC}"
pip install -e .

# 4. Generate demo samples and metadata
echo -e "${YELLOW}[5/5] Generating curated demonstration samples...${NC}"
python data/scripts/download_demo_samples.py

# 5. Verification
echo -e "${CYAN}------------------------------------------------------------${NC}"
echo -e "${YELLOW}Running verification smoke tests...${NC}"
if python scripts/smoke_test.py; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN} SatQuery AI environment setup SUCCESSFUL!${NC}"
    echo -e "${GREEN} Launch UI with:  streamlit run app/main.py${NC}"
    echo -e "${GREEN}============================================================${NC}"
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED} Verification failed. Please check the logs above.${NC}"
    echo -e "${RED}============================================================${NC}"
    exit 1
fi
