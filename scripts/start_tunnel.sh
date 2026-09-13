#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI -- Quick Cloudflare Tunnel for Live Demonstrations
# Provides a temporary public HTTPS URL for the React Mission Workstation during judging.
# No Cloudflare account required.
# ==============================================================================

set -euo pipefail

PORT="${1:-5173}"

echo "============================================================"
echo " Starting Cloudflare Quick Tunnel for SatQuery AI on port ${PORT}"
echo "============================================================"

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "ERROR: 'cloudflared' binary not found in PATH."
    echo "Please install cloudflared:"
    echo "  Linux:   sudo apt-get install cloudflared"
    echo "  macOS:   brew install cloudflared"
    echo "  Windows: winget install --id Cloudflare.cloudflared"
    echo "Or download from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
fi

echo "Connecting tunnel to http://localhost:${PORT}..."
echo "Watch output below for the *.trycloudflare.com URL:"
echo "------------------------------------------------------------"

cloudflared tunnel --url "http://localhost:${PORT}"
