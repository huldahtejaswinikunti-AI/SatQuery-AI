#!/usr/bin/env bash
set -e

echo "Running code formatting and lint checks..."
python -m flake8 satquery/ app/ tests/ --count --max-line-length=120 --statistics || true

