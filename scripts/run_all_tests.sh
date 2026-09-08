#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI -- Run All Tests (Unit, Integration, and Smoke Tests)
# SIH 2026 PS 26167 (ISRO/SAC)
# ==============================================================================

set -uo pipefail

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} Running SatQuery AI Complete Test Suite${NC}"
echo -e "${CYAN}============================================================${NC}"

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

FAILED=0

# 1. Pytest Suite
echo -e "${YELLOW}[1/3] Running pytest unit and integration tests...${NC}"
if python -m pytest tests/ -v --tb=short -q; then
    echo -e "${GREEN}--> Pytest suite passed!${NC}"
else
    echo -e "${RED}--> Pytest suite encountered failures!${NC}"
    FAILED=1
fi

# 2. Syntax & Compilation check
echo -e "${YELLOW}[2/3] Checking python syntax compilation across modules...${NC}"
if python -m py_compile app/*.py satquery/*.py scripts/*.py evaluation/*.py; then
    echo -e "${GREEN}--> Py-compile check passed!${NC}"
else
    echo -e "${RED}--> Syntax / py-compile check failed!${NC}"
    FAILED=1
fi

# 3. Pipeline Smoke Test
echo -e "${YELLOW}[3/3] Running end-to-end pipeline smoke test...${NC}"
if python scripts/smoke_test.py; then
    echo -e "${GREEN}--> Smoke test passed!${NC}"
else
    echo -e "${RED}--> Smoke test failed!${NC}"
    FAILED=1
fi

echo -e "${CYAN}============================================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN} ALL TEST SUITES PASSED CLEANLY!${NC}"
    echo -e "${CYAN}============================================================${NC}"
    exit 0
else
    echo -e "${RED} ONE OR MORE TEST SUITES FAILED! Check outputs above.${NC}"
    echo -e "${CYAN}============================================================${NC}"
    exit 1
fi
