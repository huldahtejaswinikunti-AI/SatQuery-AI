from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from satquery.utils.image_utils import extract_optical_bands

@dataclass
class CloudMaskResult:
    cloud_mask: np.ndarray
    cloud_fraction: float
    is_heavily_cloud_covered: bool

def detect_cloud_mask(optical_arr: np.ndarray, brightness_threshold: float = 0.72, saturation_threshold: float = 0.20) -> CloudMaskResult:
    bands = extract_optical_bands(optical_arr)
    r, g, b = bands["red"], bands["green"], bands["blue"]
    max_val = max(float(np.max(r)), float(np.max(g)), float(np.max(b)), 1.0)
    r_n = r / max_val if max_val > 1.0 else r
    g_n = g / max_val if max_val > 1.0 else g
    b_n = b / max_val if max_val > 1.0 else b

    brightness = (r_n + g_n + b_n) / 3.0
    rgb = np.stack([r_n, g_n, b_n], axis=-1)
    saturation = (np.max(rgb, axis=-1) - np.min(rgb, axis=-1)) / (np.max(rgb, axis=-1) + 1e-6)
    cloud_mask = (brightness >= brightness_threshold) & (saturation <= saturation_threshold)

    tot = float(optical_arr.shape[0] * optical_arr.shape[1])
    cloud_frac = float(np.sum(cloud_mask)) / tot
    return CloudMaskResult(cloud_mask=cloud_mask, cloud_fraction=round(cloud_frac, 4), is_heavily_cloud_covered=cloud_frac > 0.25)
