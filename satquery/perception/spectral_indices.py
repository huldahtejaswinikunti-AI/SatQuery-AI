from __future__ import annotations
from dataclasses import dataclass
from typing import Any
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
    """
    Computes (band_a - band_b) / (band_a + band_b).
    Where denominator is close to zero, uses eps to prevent divide-by-zero, clipped to [-1, 1].
    """
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denom = a + b
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return np.clip((a - b) / denom, -1.0, 1.0)


def compute_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Computes standard Sentinel-2 optical spectral indices (NDVI, NDWI, NDBI).

    Formulas:
        NDVI = (NIR - Red) / (NIR + Red)       (Normalized Difference Vegetation Index)
        NDWI = (Green - NIR) / (Green + NIR)   (Normalized Difference Water Index, McFeeters 1996)
        NDBI = (SWIR - NIR) / (SWIR + NIR)     (Normalized Difference Built-up Index, Zha et al. 2003)

    Divide-by-zero handling:
        Where the denominator (band_a + band_b) == 0 (or |denom| < 1e-10), the index value
        is set to 0.0 (neutral baseline, zero-masked rather than NaN) to prevent downstream
        pipeline breakages and ensure safe thresholding without NaN propagation.

    Args:
        bands: Dictionary containing Sentinel-2 band arrays with keys:
               "red", "green", "nir", "swir" (each as 2D or 3D numpy ndarray, float or uint).

    Returns:
        Dictionary with keys:
            "ndvi": ndarray of float32, bounded in [-1.0, 1.0]
            "ndwi": ndarray of float32, bounded in [-1.0, 1.0]
            "ndbi": ndarray of float32, bounded in [-1.0, 1.0]
    """
    required_keys = {"red", "green", "nir", "swir"}
    missing = required_keys - set(bands.keys())
    if missing:
        raise ValueError(f"Missing required band(s) in input dictionary: {missing}")

    red = np.asarray(bands["red"], dtype=np.float32)
    green = np.asarray(bands["green"], dtype=np.float32)
    nir = np.asarray(bands["nir"], dtype=np.float32)
    swir = np.asarray(bands["swir"], dtype=np.float32)

    def _calc_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = a + b
        numer = a - b
        # Zero-mask where denominator is zero or near zero (< 1e-10)
        safe_mask = np.abs(denom) > 1e-10
        out = np.zeros_like(denom, dtype=np.float32)
        np.divide(numer, denom, out=out, where=safe_mask)
        # Nan/inf safety and clipping to valid physical range [-1.0, 1.0]
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(out, -1.0, 1.0)

    ndvi = _calc_ratio(nir, red)
    ndwi = _calc_ratio(green, nir)
    ndbi = _calc_ratio(swir, nir)

    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi": ndbi,
    }


def compute_spectral_indices(image_arr: np.ndarray) -> SpectralIndicesResult:
    """
    High-level extractor parsing optical array into SpectralIndicesResult with masks & fractions.
    Maintained for pipeline executor backward compatibility.
    """
    bands = extract_optical_bands(image_arr)
    indices = compute_indices(bands)
    ndvi, ndwi, ndbi = indices["ndvi"], indices["ndwi"], indices["ndbi"]

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
