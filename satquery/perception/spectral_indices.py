from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from satquery.utils.config import settings
from satquery.utils.image_utils import extract_optical_bands

@dataclass
class SpectralIndicesResult:
    ndvi: np.ndarray
    ndwi: np.ndarray
    ndbi: np.ndarray
    vegetation_mask: np.ndarray
    water_mask: np.ndarray
    built_up_mask: np.ndarray
    vegetation_fraction: float
    water_fraction: float
    built_up_fraction: float

def compute_normalized_difference(band_a: np.ndarray, band_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denom = a + b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return np.clip((a - b) / denom, -1.0, 1.0)

def compute_spectral_indices(image_arr: np.ndarray) -> SpectralIndicesResult:
    bands = extract_optical_bands(image_arr)
    red, green, nir, swir = bands["red"], bands["green"], bands["nir"], bands["swir"]

    ndvi = compute_normalized_difference(nir, red)
    ndwi = compute_normalized_difference(green, nir)
    ndbi = compute_normalized_difference(swir, nir)

    thresh = settings.spectral
    vegetation_mask = ndvi >= thresh.ndvi_sparse_vegetation
    water_mask = ndwi >= thresh.ndwi_water_body
    built_up_mask = ndbi >= thresh.ndbi_built_up

    tot = float(image_arr.shape[0] * image_arr.shape[1])
    return SpectralIndicesResult(
        ndvi=ndvi, ndwi=ndwi, ndbi=ndbi,
        vegetation_mask=vegetation_mask, water_mask=water_mask, built_up_mask=built_up_mask,
        vegetation_fraction=round(float(np.sum(vegetation_mask)) / tot, 4),
        water_fraction=round(float(np.sum(water_mask)) / tot, 4),
        built_up_fraction=round(float(np.sum(built_up_mask)) / tot, 4),
    )
