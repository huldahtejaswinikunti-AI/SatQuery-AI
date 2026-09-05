#!/usr/bin/env bash
set -e

echo "Setting up SatQuery AI environment (Linux/macOS)..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

python data/scripts/download_demo_samples.py

echo "Setup completed successfully! Run 'streamlit run app/main.py'."

