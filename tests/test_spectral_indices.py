import numpy as np
import pytest
from satquery.perception.spectral_indices import (
    compute_indices,
    compute_spectral_indices,
    compute_normalized_difference,
    SpectralIndicesResult,
)


def test_normalized_difference_basic():
    a = np.array([100.0, 50.0], dtype=np.float32)
    b = np.array([50.0, 50.0], dtype=np.float32)
    res = compute_normalized_difference(a, b)
    assert pytest.approx(res[0], rel=1e-3) == (100 - 50) / 150
    assert res[1] == 0.0


def test_compute_indices_known_values():
    """
    Test compute_indices with known mathematical outputs:
    NDVI = (NIR - Red) / (NIR + Red)
    NDWI = (Green - NIR) / (Green + NIR)
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """
    h, w = 4, 4
    # NIR=0.8, Red=0.2 => NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.60
    # Green=0.6, NIR=0.2 (wait, NIR is 0.8 here):
    # NDWI with Green=0.4, NIR=0.8 => (0.4 - 0.8) / (0.4 + 0.8) = -0.4 / 1.2 = -0.3333
    # NDBI with SWIR=0.4, NIR=0.8 => (0.4 - 0.8) / (0.4 + 0.8) = -0.4 / 1.2 = -0.3333
    bands = {
        "red": np.full((h, w), 0.2, dtype=np.float32),
        "green": np.full((h, w), 0.4, dtype=np.float32),
        "nir": np.full((h, w), 0.8, dtype=np.float32),
        "swir": np.full((h, w), 0.4, dtype=np.float32),
    }
    indices = compute_indices(bands)

    assert "ndvi" in indices
    assert "ndwi" in indices
    assert "ndbi" in indices

    expected_ndvi = (0.8 - 0.2) / (0.8 + 0.2)  # 0.6
    expected_ndwi = (0.4 - 0.8) / (0.4 + 0.8)  # -1/3
    expected_ndbi = (0.4 - 0.8) / (0.4 + 0.8)  # -1/3

    assert np.allclose(indices["ndvi"], expected_ndvi, atol=1e-5)
    assert np.allclose(indices["ndwi"], expected_ndwi, atol=1e-5)
    assert np.allclose(indices["ndbi"], expected_ndbi, atol=1e-5)


def test_compute_indices_zero_division_guard():
    """
    Test divide-by-zero guard: where NIR + Red == 0, NDVI must return 0.0 (not NaN).
    """
    h, w = 3, 3
    bands = {
        "red": np.zeros((h, w), dtype=np.float32),
        "green": np.zeros((h, w), dtype=np.float32),
        "nir": np.zeros((h, w), dtype=np.float32),
        "swir": np.zeros((h, w), dtype=np.float32),
    }
    indices = compute_indices(bands)

    assert not np.any(np.isnan(indices["ndvi"]))
    assert not np.any(np.isnan(indices["ndwi"]))
    assert not np.any(np.isnan(indices["ndbi"]))

    assert np.all(indices["ndvi"] == 0.0)
    assert np.all(indices["ndwi"] == 0.0)
    assert np.all(indices["ndbi"] == 0.0)


def test_compute_indices_bounded_range():
    """
    Test that indices never exceed physical boundary [-1.0, 1.0].
    """
    bands = {
        "red": np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32),
        "green": np.array([[0.5, 1.0], [0.0, 5.0]], dtype=np.float32),
        "nir": np.array([[0.0, 1.0], [10.0, 0.0]], dtype=np.float32),
        "swir": np.array([[0.2, 0.5], [1.0, 20.0]], dtype=np.float32),
    }
    indices = compute_indices(bands)

    for name in ["ndvi", "ndwi", "ndbi"]:
        arr = indices[name]
        assert np.all(arr >= -1.0)
        assert np.all(arr <= 1.0)


def test_compute_indices_missing_band_raises():
    with pytest.raises(ValueError, match="Missing required band"):
        compute_indices({"red": np.zeros((2, 2)), "nir": np.ones((2, 2))})


def test_backward_compat_compute_spectral_indices():
    h, w = 16, 16
    red = np.full((h, w), 0.1, dtype=np.float32)
    green = np.full((h, w), 0.2, dtype=np.float32)
    nir = np.full((h, w), 0.8, dtype=np.float32)
    swir = np.full((h, w), 0.1, dtype=np.float32)
    img = np.stack([red, green, red, nir, swir], axis=-1)
    res = compute_spectral_indices(img)

    assert isinstance(res, SpectralIndicesResult)
    assert np.mean(res.ndvi) > 0.7
    assert res.vegetation_fraction > 0.9
    assert res.water_fraction == 0.0
