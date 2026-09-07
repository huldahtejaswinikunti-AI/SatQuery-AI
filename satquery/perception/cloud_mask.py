from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from satquery.utils.image_utils import extract_optical_bands

@dataclass
class CloudMaskResult:
    cloud_mask: np.ndarray
    cloud_fraction: float
    is_heavily_cloud_covered: bool


def compute_cloud_mask(
    bands: dict[str, np.ndarray] | np.ndarray,
    qa60: np.ndarray | None = None,
    brightness_threshold: float = 0.35,
    saturation_threshold: float = 0.20,
) -> np.ndarray:
    """
    Detects cloud-obscured pixels for optical satellite imagery (Sentinel-2).

    Supports two detection pathways:
    1. Direct QA60 bitmask parsing (Sentinel-2 Quality Assessment Band):
       - Bit 10 (1024 / 0x400): Opaque clouds
       - Bit 11 (2048 / 0x800): Cirrus clouds
       Returns True where either bit 10 or bit 11 is flagged.
    2. Spectral Heuristic (when QA60 is unavailable):
       - Evaluates high broadband top-of-atmosphere/surface reflectance and low color saturation.
       - Clouds exhibit high uniform reflectance across visible bands (Red, Green, Blue) and
         distinctly low color saturation compared to high-albedo terrestrial features (like sand or salt flats).
       - Brightness = (Red + Green + Blue) / 3.0 >= brightness_threshold
       - Saturation = (max(RGB) - min(RGB)) / (max(RGB) + eps) <= saturation_threshold

    Args:
        bands: Either a dictionary containing "red", "green", "blue" (float32 reflectance 0-1
               or scaled DN), or a 3D/2D optical image array.
        qa60: Optional 2D integer ndarray containing Sentinel-2 QA60 bitmask.
        brightness_threshold: Reflectance/intensity cutoff for cloud candidates (default: 0.35).
        saturation_threshold: Color saturation cutoff (default: 0.20).

    Returns:
        boolean ndarray of shape (H, W), True where cloud-obscured, False where clear.
    """
    if qa60 is not None:
        qa = np.asarray(qa60, dtype=np.uint32)
        # Bit 10: Opaque cloud, Bit 11: Cirrus cloud
        opaque_cloud = (qa & (1 << 10)) != 0
        cirrus_cloud = (qa & (1 << 11)) != 0
        return (opaque_cloud | cirrus_cloud).astype(bool)

    # Spectral heuristic based on RGB reflectance
    if isinstance(bands, dict):
        # Extract R, G, B from dictionary
        r = np.asarray(bands.get("red", bands.get("r")), dtype=np.float32)
        g = np.asarray(bands.get("green", bands.get("g")), dtype=np.float32)
        b = np.asarray(bands.get("blue", bands.get("b")), dtype=np.float32)
    else:
        extracted = extract_optical_bands(np.asarray(bands))
        r = extracted["red"]
        g = extracted["green"]
        b = extracted["blue"]

    # Normalize to [0.0, 1.0] if data is 8-bit or 12-bit/16-bit DN
    max_val = max(float(np.nanmax(r)), float(np.nanmax(g)), float(np.nanmax(b)), 1.0)
    r_n = r / max_val if max_val > 1.0 else r
    g_n = g / max_val if max_val > 1.0 else g
    b_n = b / max_val if max_val > 1.0 else b

    brightness = (r_n + g_n + b_n) / 3.0
    rgb = np.stack([r_n, g_n, b_n], axis=-1)
    rgb_max = np.max(rgb, axis=-1)
    rgb_min = np.min(rgb, axis=-1)
    saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)

    # Cloud: High reflectance AND Low saturation
    cloud_mask = (brightness >= brightness_threshold) & (saturation <= saturation_threshold)
    return cloud_mask.astype(bool)


def detect_cloud_mask(
    optical_arr: np.ndarray,
    brightness_threshold: float = 0.72,
    saturation_threshold: float = 0.20
) -> CloudMaskResult:
    """
    Detects cloud mask and summarizes cloud fraction.
    Maintained for pipeline executor backward compatibility.
    """
    cloud_mask = compute_cloud_mask(
        optical_arr,
        qa60=None,
        brightness_threshold=brightness_threshold,
        saturation_threshold=saturation_threshold
    )

    tot = float(optical_arr.shape[0] * optical_arr.shape[1])
    cloud_frac = float(np.sum(cloud_mask)) / tot
    return CloudMaskResult(
        cloud_mask=cloud_mask,
        cloud_fraction=round(cloud_frac, 4),
        is_heavily_cloud_covered=cloud_frac > 0.25
    )
