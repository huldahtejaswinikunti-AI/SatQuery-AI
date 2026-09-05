#!/usr/bin/env bash
set -e

echo "Running full test suite for SatQuery AI..."
pytest tests/ -v --tb=short

