# ==============================================================================
# SatQuery AI -- Quick Cloudflare Tunnel for Live Demonstrations (PowerShell)
# Provides a temporary public HTTPS URL for the React Mission Workstation during judging.
# No Cloudflare account required.
# ==============================================================================

param(
    [int]$Port = 5173
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Starting Cloudflare Quick Tunnel for SatQuery AI on port $Port" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'cloudflared' command not found in PATH." -ForegroundColor Red
    Write-Host "Please install cloudflared using winget:" -ForegroundColor Yellow
    Write-Host "  winget install --id Cloudflare.cloudflared" -ForegroundColor White
    Write-Host "Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
}

Write-Host "Connecting tunnel to http://localhost:$Port..." -ForegroundColor Green
Write-Host "Watch output below for the *.trycloudflare.com URL:" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------"

& cloudflared tunnel --url "http://localhost:$Port"
