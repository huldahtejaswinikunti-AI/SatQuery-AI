import numpy as np
import pytest
from satquery.perception.spectral_indices import compute_spectral_indices, compute_normalized_difference

def test_normalized_difference():
    a = np.array([100.0, 50.0], dtype=np.float32)
    b = np.array([50.0, 50.0], dtype=np.float32)
    res = compute_normalized_difference(a, b)
    assert pytest.approx(res[0], rel=1e-3) == (100 - 50) / 150
    assert res[1] == 0.0

def test_compute_spectral_indices():
    h, w = 32, 32
    red = np.full((h, w), 0.1, dtype=np.float32)
    green = np.full((h, w), 0.2, dtype=np.float32)
    nir = np.full((h, w), 0.8, dtype=np.float32)
    swir = np.full((h, w), 0.1, dtype=np.float32)
    img = np.stack([red, green, red, nir, swir], axis=-1)
    res = compute_spectral_indices(img)
    assert np.mean(res.ndvi) > 0.7
    assert res.vegetation_fraction > 0.9
