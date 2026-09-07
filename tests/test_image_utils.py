"""Tests for satquery.utils.image_utils — pure array processing."""

from __future__ import annotations

import numpy as np
import pytest

from satquery.utils.image_utils import (
    extract_bands,
    normalize,
    resize,
    to_rgb_preview,
)


class TestResize:
    """Array resizing."""

    def test_resize_2d(self):
        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[8:24, 8:24] = 255
        out = resize(arr, 64, 64)
        assert out.shape == (64, 64)

    def test_resize_3d_channel_first(self):
        arr = np.zeros((3, 32, 32), dtype=np.uint8)
        arr[:, 8:24, 8:24] = 255
        out = resize(arr, 64, 64)
        assert out.shape == (3, 64, 64)

    def test_resize_invalid_dim(self):
        arr = np.zeros((1, 2, 3, 4))
        with pytest.raises(ValueError, match="Expected 2-D or 3-D"):
            resize(arr, 10, 10)


class TestNormalize:
    """Normalization modes."""

    def test_minmax(self):
        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        out = normalize(arr, method="minmax")
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)
        assert out.dtype == np.float32

    def test_minmax_constant(self):
        arr = np.full((10, 10), 5.0)
        out = normalize(arr, method="minmax")
        assert np.all(out == 0.0)

    def test_standard(self):
        arr = np.arange(100, dtype=np.float32)
        out = normalize(arr, method="standard")
        assert out.mean() == pytest.approx(0.0, abs=1e-5)
        assert out.std() == pytest.approx(1.0, abs=1e-5)

    def test_sentinel2_reflectance(self):
        arr = np.array([5000, 10000], dtype=np.float32)
        out = normalize(arr, method="sentinel2_reflectance")
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize(np.zeros((2, 2)), method="invalid_mode")


class TestExtractBands:
    """Band extraction from dictionary."""

    def test_extract_present_bands(self):
        b1 = np.ones((10, 10), dtype=np.float32) * 1
        b2 = np.ones((10, 10), dtype=np.float32) * 2
        b3 = np.ones((10, 10), dtype=np.float32) * 3
        bands = {"b1": b1, "b2": b2, "b3": b3}

        stacked = extract_bands(bands, ["b3", "b1"])
        assert stacked.shape == (2, 10, 10)
        assert np.all(stacked[0] == 3)
        assert np.all(stacked[1] == 1)

    def test_extract_missing_band(self):
        bands = {"b1": np.zeros((5, 5))}
        with pytest.raises(KeyError, match="Band 'b2' not found"):
            extract_bands(bands, ["b1", "b2"])


class TestRGBPreview:
    """RGB preview generation."""

    def test_preview_from_numbered_bands(self):
        bands = {
            "band_4": np.random.randint(0, 10000, (32, 32), dtype=np.uint16),
            "band_3": np.random.randint(0, 10000, (32, 32), dtype=np.uint16),
            "band_2": np.random.randint(0, 10000, (32, 32), dtype=np.uint16),
        }
        preview = to_rgb_preview(bands)
        assert preview.shape == (32, 32, 3)
        assert preview.dtype == np.uint8

    def test_preview_from_named_bands(self):
        bands = {
            "red": np.random.randint(0, 256, (32, 32), dtype=np.uint8),
            "green": np.random.randint(0, 256, (32, 32), dtype=np.uint8),
            "blue": np.random.randint(0, 256, (32, 32), dtype=np.uint8),
        }
        preview = to_rgb_preview(bands)
        assert preview.shape == (32, 32, 3)
        assert preview.dtype == np.uint8
