"""Contract test for API response spectral interpretation and model calibration (Bug #1).

Verifies that:
1. /api/analyze includes ndvi_label, ndwi_label, ndbi_label in its response payload.
2. The returned labels exactly equal the server-side interpret_ndvi(ndvi_mean),
   interpret_ndwi(ndwi_mean), and interpret_ndbi(ndbi_mean) values.
3. The response includes model_calibration ('calibrated' | 'untrained_fallback').
"""

import pytest
from fastapi.testclient import TestClient

from app.api_server import app
from satquery.perception.spectral_interpretation import (
    interpret_ndvi,
    interpret_ndwi,
    interpret_ndbi,
)


@pytest.fixture
def api_client():
    return TestClient(app)


def test_analyze_response_includes_spectral_labels_contract(api_client):
    """Assert /api/analyze attaches backend-computed spectral labels matching spectral_interpretation.py."""
    payload = {
        "mode": "earth",
        "files": ["single_optical/sample_agri_fields.png"],
        "query": "Analyze vegetation and spectral indices.",
    }

    response = api_client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"

    data = response.json()

    # 1. Assert model_calibration is present at top level
    assert "model_calibration" in data
    assert data["model_calibration"] in ("calibrated", "untrained_fallback")

    # 2. Assert spectral_summary is present and contains interpreted labels
    spectral = data.get("spectral_summary") or data.get("verified_facts", {}).get("spectral_summary")
    assert spectral is not None, "spectral_summary must be present in response"

    assert "ndvi_mean" in spectral
    assert "ndvi_label" in spectral
    assert "ndwi_mean" in spectral
    assert "ndwi_label" in spectral
    assert "ndbi_mean" in spectral
    assert "ndbi_label" in spectral

    # 3. Contract: labels MUST strictly equal the output of spectral_interpretation functions
    ndvi_mean = float(spectral["ndvi_mean"])
    ndwi_mean = float(spectral["ndwi_mean"])
    ndbi_mean = float(spectral["ndbi_mean"])

    assert spectral["ndvi_label"] == interpret_ndvi(ndvi_mean)
    assert spectral["ndwi_label"] == interpret_ndwi(ndwi_mean)
    assert spectral["ndbi_label"] == interpret_ndbi(ndbi_mean)


def test_health_check_includes_calibration_status(api_client):
    """Assert /api/health exposes classifier_calibration and dynamic readiness."""
    response = api_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] in ("READY", "UNCALIBRATED")
    assert "classifier_calibration" in data
    assert data["classifier_calibration"] in ("calibrated", "untrained_fallback")
