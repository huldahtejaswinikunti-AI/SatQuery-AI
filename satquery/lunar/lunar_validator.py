"""Lightweight validator for Lunar imagery (Chandrayaan-2 OHRC / TMC-2).

Checks file validity, raster channels, and query presence without assuming
Earth-observation optical/SAR band configurations or multi-image pairings.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def validate_lunar_input(
    images: list[np.ndarray],
    metas: list[dict[str, Any]],
    query: str,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate a lunar input item.

    Parameters
    ----------
    images : list[np.ndarray]
        List containing exactly one lunar raster image.
    metas : list[dict]
        Metadata dict for each image.
    query : str
        User natural-language query.

    Returns
    -------
    tuple[bool, str | None, dict]
        (is_valid, error_message, normalized_meta)
    """
    if not images:
        return False, "No lunar imagery provided. Please select a Chandrayaan-2 demo sample or upload an image.", {}

    if len(images) > 1:
        return False, f"Lunar mode currently analyzes one product at a time (received {len(images)} images).", {}

    if not query or not query.strip():
        return False, "Query cannot be empty. Please ask a question about craters, regolith, or lunar terrain.", {}

    img = images[0]
    if not isinstance(img, np.ndarray):
        return False, f"Expected image as numpy array, got {type(img).__name__}.", {}

    if img.ndim < 2 or img.ndim > 3:
        return False, f"Invalid image dimensions {img.shape}. Expected 2D or 3D raster.", {}

    h, w = img.shape[:2]
    if h < 16 or w < 16:
        return False, f"Image dimensions too small ({w}x{h}). Minimum size is 16x16.", {}

    m = dict(metas[0]) if metas else {}
    m.setdefault("width", w)
    m.setdefault("height", h)
    m.setdefault("channels", 1 if img.ndim == 2 else img.shape[2])
    m.setdefault("sensor", "Chandrayaan-2 OHRC / TMC-2")
    m.setdefault("target_body", "Moon")

    return True, None, m
