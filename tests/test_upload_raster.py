"""Test suite for custom raster upload endpoint and analysis integration."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.api_server import app, UPLOAD_DIR


@pytest.fixture
def client():
    return TestClient(app)


def _create_test_png_bytes(width=64, height=64, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_upload_raster_success(client):
    """Assert /api/upload accepts valid PNG/GeoTIFF raster, saves it, and extracts metadata."""
    png_bytes = _create_test_png_bytes(128, 96, (80, 120, 160))
    files = [("files", ("test_custom_raster.png", png_bytes, "image/png"))]

    response = client.post("/api/upload?domain=earth", files=files)
    assert response.status_code == 200, f"Upload failed: {response.text}"

    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    item = data["items"][0]

    assert item["filename"] == "test_custom_raster.png"
    assert item["file"].startswith("uploads/")
    assert item["width"] == 128
    assert item["height"] == 96
    assert item["band_count"] == 3
    assert (UPLOAD_DIR / "test_custom_raster.png").exists()


def test_upload_and_analyze_integration(client):
    """Assert uploaded raster can be directly analyzed via /api/analyze with custom query."""
    png_bytes = _create_test_png_bytes(64, 64, (40, 180, 90))
    files = [("files", ("test_custom_agri.png", png_bytes, "image/png"))]

    upload_res = client.post("/api/upload?domain=earth", files=files)
    assert upload_res.status_code == 200
    uploaded_file_path = upload_res.json()["items"][0]["file"]

    # Execute analyze on the uploaded raster
    analyze_payload = {
        "mode": "earth",
        "files": [uploaded_file_path],
        "query": "Describe the dominant land use and spectral characteristics in this custom scene.",
    }

    res = client.post("/api/analyze", json=analyze_payload)
    assert res.status_code == 200, f"Analyze failed: {res.text}"
    result = res.json()
    assert "answer" in result
    assert "confidence" in result
    assert result["confidence_tag"] != "error"
    assert "report_markdown" in result


def test_upload_invalid_extension(client):
    """Assert /api/upload rejects non-raster formats."""
    dummy_bytes = b"Hello, this is not a raster satellite image."
    files = [("files", ("sample.txt", dummy_bytes, "text/plain"))]

    response = client.post("/api/upload?domain=earth", files=files)
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]
