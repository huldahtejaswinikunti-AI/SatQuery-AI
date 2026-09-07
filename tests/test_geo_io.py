"""Tests for satquery.utils.geo_io — image loading across formats."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.utils.geo_io import load_image


class TestLoadGeoTIFF:
    """GeoTIFF loading via rasterio."""

    def test_3band_optical(self, sample_geotiff_3band):
        result = load_image(str(sample_geotiff_3band))

        assert "bands" in result
        assert "metadata" in result

        bands = result["bands"]
        meta = result["metadata"]

        assert len(bands) == 3
        assert set(bands.keys()) == {"band_1", "band_2", "band_3"}

        for arr in bands.values():
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (64, 64)

        assert meta["band_count"] == 3
        assert meta["crs"] is not None
        assert meta["transform"] is not None
        assert meta["bounds"] is not None
        assert meta["format"] == "GeoTIFF"
        assert meta["width"] == 64
        assert meta["height"] == 64

    def test_1band_sar(self, sample_geotiff_1band):
        result = load_image(str(sample_geotiff_1band))

        bands = result["bands"]
        meta = result["metadata"]

        assert len(bands) == 1
        assert "band_1" in bands
        assert bands["band_1"].shape == (64, 64)
        assert meta["band_count"] == 1


class TestLoadPNG:
    """PNG loading via Pillow."""

    def test_rgb_png(self, sample_png):
        result = load_image(str(sample_png))

        bands = result["bands"]
        meta = result["metadata"]

        assert set(bands.keys()) == {"red", "green", "blue"}
        for arr in bands.values():
            assert arr.shape == (64, 64)

        assert meta["crs"] is None
        assert meta["transform"] is None
        assert meta["format"] == "PNG"
        assert meta["band_count"] == 3

    def test_grayscale_png(self, sample_grayscale_png):
        result = load_image(str(sample_grayscale_png))

        bands = result["bands"]
        assert "gray" in bands
        assert result["metadata"]["band_count"] == 1

    def test_jpeg(self, sample_jpeg):
        result = load_image(str(sample_jpeg))
        assert result["metadata"]["format"] == "JPEG"
        assert "red" in result["bands"]


class TestLoadErrors:
    """Error handling for missing or unsupported files."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/to/image.tif")

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.bmp"
        bad_file.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image(str(bad_file))
