"""
Image preprocessing and band extraction.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

def normalize_percentile(arr: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    valid_mask = np.isfinite(arr)
    if not np.any(valid_mask):
        return np.zeros_like(arr, dtype=np.float32)
    p_low = np.percentile(arr[valid_mask], lower_pct)
    p_high = np.percentile(arr[valid_mask], upper_pct)
    if p_high <= p_low:
        p_high = p_low + 1e-6
    return np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0)

def to_display_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        norm = normalize_percentile(arr)
        return (np.stack([norm, norm, norm], axis=-1) * 255).astype(np.uint8)
    channels = arr.shape[-1]
    if channels == 1:
        norm = normalize_percentile(arr[..., 0])
        return (np.stack([norm, norm, norm], axis=-1) * 255).astype(np.uint8)
    elif channels == 2:
        vv = normalize_percentile(arr[..., 0])
        vh = normalize_percentile(arr[..., 1])
        ratio = normalize_percentile(arr[..., 0] / (arr[..., 1] + 1e-6))
        return (np.stack([vv, vh, ratio], axis=-1) * 255).astype(np.uint8)
    elif channels in (3, 4):
        if arr.dtype == np.uint8:
            return arr[..., :3]
        r = normalize_percentile(arr[..., 0])
        g = normalize_percentile(arr[..., 1])
        b = normalize_percentile(arr[..., 2])
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    elif channels >= 10:
        r = normalize_percentile(arr[..., 3])
        g = normalize_percentile(arr[..., 2])
        b = normalize_percentile(arr[..., 1])
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    else:
        r = normalize_percentile(arr[..., 0])
        g = normalize_percentile(arr[..., 1])
        b = normalize_percentile(arr[..., 2 if channels > 2 else 0])
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def extract_optical_bands(arr: np.ndarray) -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    if arr.ndim == 2:
        for b in ["red", "green", "blue", "nir", "swir"]:
            bands[b] = arr.astype(np.float32)
        return bands
    channels = arr.shape[-1]
    if channels >= 12:
        bands["blue"] = arr[..., 1].astype(np.float32)
        bands["green"] = arr[..., 2].astype(np.float32)
        bands["red"] = arr[..., 3].astype(np.float32)
        bands["nir"] = arr[..., 7].astype(np.float32)
        bands["swir"] = arr[..., 10].astype(np.float32)
    elif channels >= 4:
        bands["red"] = arr[..., 0].astype(np.float32)
        bands["green"] = arr[..., 1].astype(np.float32)
        bands["blue"] = arr[..., 2].astype(np.float32)
        bands["nir"] = arr[..., 3].astype(np.float32)
        bands["swir"] = (arr[..., 0] * 0.7 + arr[..., 3] * 0.3).astype(np.float32)
    else:
        bands["red"] = arr[..., 0].astype(np.float32)
        bands["green"] = arr[..., 1].astype(np.float32)
        bands["blue"] = arr[..., 2].astype(np.float32)
        bands["nir"] = (arr[..., 1] * 1.4 - arr[..., 0] * 0.4).astype(np.float32)
        bands["swir"] = (arr[..., 0] * 1.1).astype(np.float32)
    return bands

def extract_sar_bands(arr: np.ndarray) -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    if arr.ndim == 2:
        bands["vv"] = arr.astype(np.float32)
        bands["vh"] = (arr * 0.5).astype(np.float32)
    elif arr.shape[-1] == 1:
        bands["vv"] = arr[..., 0].astype(np.float32)
        bands["vh"] = (arr[..., 0] * 0.5).astype(np.float32)
    else:
        bands["vv"] = arr[..., 0].astype(np.float32)
        bands["vh"] = arr[..., 1].astype(np.float32)
    return bands
