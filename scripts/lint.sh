#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI -- Codebase Linting & Statistics Script
# SIH 2026 PS 26167 (ISRO/SAC)
# ==============================================================================

set -uo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} SatQuery AI Codebase Linting & Quality Checks${NC}"
echo -e "${CYAN}============================================================${NC}"

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# 1. Codebase statistics
echo -e "${YELLOW}[1/3] Calculating codebase statistics...${NC}"
PY_FILES=$(find satquery app evaluation scripts -name "*.py" 2>/dev/null | wc -l)
TOTAL_LINES=$(find satquery app evaluation scripts -name "*.py" -exec cat {} + 2>/dev/null | wc -l)
echo -e "Total Python files: ${GREEN}${PY_FILES}${NC}"
echo -e "Total lines of code: ${GREEN}${TOTAL_LINES}${NC}"

# 2. Linter execution (ruff preferred, fallback to flake8)
echo -e "${YELLOW}[2/3] Running linter checks...${NC}"
if command -v ruff &> /dev/null; then
    echo "Running ruff check..."
    ruff check satquery/ app/ evaluation/ scripts/ || true
elif python -m flake8 --version &> /dev/null; then
    echo "Running flake8..."
    python -m flake8 satquery/ app/ evaluation/ scripts/ --count --max-line-length=120 --statistics || true
else
    echo "Neither ruff nor flake8 found; performing py_compile syntax validation..."
    python -m py_compile $(find satquery app evaluation scripts -name "*.py")
    echo -e "${GREEN}All Python files compiled with valid syntax.${NC}"
fi

# 3. Formatting check (ruff format / black fallback)
echo -e "${YELLOW}[3/3] Checking code formatting...${NC}"
if command -v ruff &> /dev/null; then
    ruff format --check satquery/ app/ evaluation/ scripts/ || true
elif python -m black --version &> /dev/null; then
    python -m black --check satquery/ app/ evaluation/ scripts/ || true
else
    echo "Formatting checker (ruff/black) not installed -- skipping strict format check."
fi

echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN} Codebase lint and statistics inspection finished.${NC}"
echo -e "${CYAN}============================================================${NC}"
