# ==============================================================================
# SatQuery AI -- Run All Tests (Windows PowerShell)
# SIH 2026 PS 26167 (ISRO/SAC)
# ==============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Running SatQuery AI Complete Test Suite (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$Failed = 0

# 1. Pytest Suite
Write-Host "[1/3] Running pytest unit and integration tests..." -ForegroundColor Yellow
python -m pytest tests/ -v --tb=short -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "--> Pytest suite encountered failures!" -ForegroundColor Red
    $Failed = 1
} else {
    Write-Host "--> Pytest suite passed!" -ForegroundColor Green
}

# 2. Syntax check
Write-Host "[2/3] Checking python syntax compilation across modules..." -ForegroundColor Yellow
$pyFiles = Get-ChildItem -Path "app\*.py", "scripts\*.py", "evaluation\*.py" | Select-Object -ExpandProperty FullName
python -m py_compile $pyFiles
if ($LASTEXITCODE -ne 0) {
    Write-Host "--> Syntax / py-compile check failed!" -ForegroundColor Red
    $Failed = 1
} else {
    Write-Host "--> Py-compile check passed!" -ForegroundColor Green
}

# 3. Smoke test
Write-Host "[3/3] Running end-to-end pipeline smoke test..." -ForegroundColor Yellow
python scripts\smoke_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "--> Smoke test failed!" -ForegroundColor Red
    $Failed = 1
} else {
    Write-Host "--> Smoke test passed!" -ForegroundColor Green
}

Write-Host "============================================================" -ForegroundColor Cyan
if ($Failed -eq 0) {
    Write-Host " ALL TEST SUITES PASSED CLEANLY!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host " ONE OR MORE TEST SUITES FAILED! Check outputs above." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 1
}
