# PowerShell one-shot setup script for Windows
Write-Host "Setting up SatQuery AI environment on Windows..." -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& .venv\Scripts\Activate.ps1

Write-Host "Installing requirements..."
pip install -r requirements.txt
pip install -e .

Write-Host "Generating curated demonstration samples..."
python data\scripts\download_demo_samples.py

Write-Host "Setup complete! Run 'streamlit run app/main.py' to start the application." -ForegroundColor Green

