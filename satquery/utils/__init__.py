"""Shared low-level utilities for SatQuery AI."""

from satquery.utils.config import (
    DEVICE,
    MAX_IMAGE_DIM,
    NDVI_THRESHOLD,
    NDWI_THRESHOLD,
    OPTICAL_BAND_COUNT_MIN,
    PHRASING_MODEL_ID,
    ROUTER_CONFIDENCE_THRESHOLD,
    SAR_BAND_COUNT_MAX,
    SPECIALIST_MAX_RETRIES,
)
from satquery.utils.geo_io import load_image
from satquery.utils.image_utils import (
    extract_bands,
    normalize,
    resize,
    to_rgb_preview,
)

try:
    from satquery.utils.overlay import (
        create_change_overlay,
        create_mask_overlay,
        create_side_by_side,
        draw_overlay,
    )
except ImportError:
    create_change_overlay = None  # type: ignore
    create_mask_overlay = None  # type: ignore
    create_side_by_side = None  # type: ignore
    draw_overlay = None  # type: ignore

__all__ = [
    "load_image",
    "resize",
    "normalize",
    "extract_bands",
    "to_rgb_preview",
    "create_mask_overlay",
    "create_change_overlay",
    "create_side_by_side",
    "draw_overlay",
    "DEVICE",
    "MAX_IMAGE_DIM",
    "NDVI_THRESHOLD",
    "NDWI_THRESHOLD",
    "OPTICAL_BAND_COUNT_MIN",
    "SAR_BAND_COUNT_MAX",
    "PHRASING_MODEL_ID",
    "ROUTER_CONFIDENCE_THRESHOLD",
    "SPECIALIST_MAX_RETRIES",
]
