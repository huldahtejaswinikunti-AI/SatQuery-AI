# SatQuery AI — Deployment Guide

## 1. Local Laptop Deployment (Recommended for Hackathon Demo)

```bash
# Clone repository
git clone <repo-url> SatQuery-AI && cd SatQuery-AI

# Create virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Launch Streamlit web application
streamlit run app/main.py
```

## 2. Running Automated Tests

```bash
pytest tests/ -v
```

## 3. Hugging Face Spaces Deployment

1. Create a new Streamlit Space on Hugging Face.
2. Push repository files to the Space git remote.
3. Configure `SATQUERY_USE_MOCK_FALLBACKS=true` for free CPU tier, or attach T4 GPU for full 4-bit GeoChat inference.

