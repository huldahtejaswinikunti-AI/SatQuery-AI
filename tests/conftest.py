"""Shared pytest fixtures for SatQuery AI tests.

Generates synthetic GeoTIFFs, PNGs, and sample queries in temporary
directories so tests run without real satellite data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture directory (tests/fixtures/) — created once per session
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_dir():
    """Create the fixtures directory if it doesn't exist."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic GeoTIFF generator
# ---------------------------------------------------------------------------


def _create_geotiff(
    path: Path,
    *,
    width: int = 64,
    height: int = 64,
    band_count: int = 3,
    crs_epsg: int = 4326,
    dtype: str = "uint16",
) -> Path:
    """Write a minimal GeoTIFF with random data."""
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(
        west=77.0, south=12.0, east=77.1, north=12.1,
        width=width, height=height,
    )

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": band_count,
        "crs": f"EPSG:{crs_epsg}",
        "transform": transform,
    }

    rng = np.random.default_rng(42)
    with rasterio.open(str(path), "w", **profile) as dst:
        for band_idx in range(1, band_count + 1):
            data = rng.integers(0, 10000, size=(height, width), dtype=np.uint16)
            dst.write(data, band_idx)

    return path


@pytest.fixture(scope="session")
def sample_geotiff_3band(ensure_fixtures_dir) -> Path:
    """3-band (optical) synthetic GeoTIFF."""
    p = FIXTURES_DIR / "optical_3band.tif"
    if not p.exists():
        _create_geotiff(p, band_count=3)
    return p


@pytest.fixture(scope="session")
def sample_geotiff_1band(ensure_fixtures_dir) -> Path:
    """1-band (SAR) synthetic GeoTIFF."""
    p = FIXTURES_DIR / "sar_1band.tif"
    if not p.exists():
        _create_geotiff(p, band_count=1)
    return p


@pytest.fixture(scope="session")
def sample_geotiff_different_crs(ensure_fixtures_dir) -> Path:
    """3-band GeoTIFF with a different CRS (EPSG:32643 — UTM zone 43N)."""
    p = FIXTURES_DIR / "optical_utm43n.tif"
    if not p.exists():
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(
            west=500000, south=1300000, east=510000, north=1310000,
            width=64, height=64,
        )
        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "width": 64,
            "height": 64,
            "count": 3,
            "crs": "EPSG:32643",
            "transform": transform,
        }
        rng = np.random.default_rng(99)
        with rasterio.open(str(p), "w", **profile) as dst:
            for i in range(1, 4):
                dst.write(rng.integers(0, 10000, (64, 64), dtype=np.uint16), i)
    return p


# ---------------------------------------------------------------------------
# PNG / JPEG fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_png(ensure_fixtures_dir) -> Path:
    """Simple 64x64 RGB PNG."""
    from PIL import Image

    p = FIXTURES_DIR / "sample.png"
    if not p.exists():
        rng = np.default_rng(7)
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(p))
    return p


@pytest.fixture(scope="session")
def sample_jpeg(ensure_fixtures_dir) -> Path:
    """Simple 64x64 RGB JPEG."""
    from PIL import Image

    p = FIXTURES_DIR / "sample.jpg"
    if not p.exists():
        rng = np.default_rng(11)
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(p))
    return p


@pytest.fixture(scope="session")
def sample_grayscale_png(ensure_fixtures_dir) -> Path:
    """64x64 grayscale PNG."""
    from PIL import Image

    p = FIXTURES_DIR / "gray.png"
    if not p.exists():
        rng = np.default_rng(13)
        arr = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(str(p))
    return p


# ---------------------------------------------------------------------------
# Helper dicts (simulate geo_io.load_image output for validator tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def optical_image_meta(sample_geotiff_3band) -> dict:
    """Metadata dict for a 3-band optical image (validator input shape)."""
    return {
        "path": str(sample_geotiff_3band),
        "format": "GeoTIFF",
        "band_count": 3,
        "crs": 4326,
        "transform": (0.0015625, 0.0, 77.0, 0.0, -0.0015625, 12.1),
        "bounds": {"left": 77.0, "bottom": 12.0, "right": 77.1, "top": 12.1},
        "width": 64,
        "height": 64,
        "timestamp": "2025-06-01T10:00:00+05:30",
    }


@pytest.fixture()
def sar_image_meta(sample_geotiff_1band) -> dict:
    """Metadata dict for a 1-band SAR image (validator input shape)."""
    return {
        "path": str(sample_geotiff_1band),
        "format": "GeoTIFF",
        "band_count": 1,
        "crs": 4326,
        "transform": (0.0015625, 0.0, 77.0, 0.0, -0.0015625, 12.1),
        "bounds": {"left": 77.0, "bottom": 12.0, "right": 77.1, "top": 12.1},
        "width": 64,
        "height": 64,
    }


@pytest.fixture()
def optical_image_meta_different_crs(sample_geotiff_different_crs) -> dict:
    """Metadata dict for optical image with UTM 43N CRS."""
    return {
        "path": str(sample_geotiff_different_crs),
        "format": "GeoTIFF",
        "band_count": 3,
        "crs": 32643,
        "transform": (156.25, 0.0, 500000, 0.0, -156.25, 1310000),
        "bounds": {"left": 500000, "bottom": 1300000, "right": 510000, "top": 1310000},
        "width": 64,
        "height": 64,
    }
