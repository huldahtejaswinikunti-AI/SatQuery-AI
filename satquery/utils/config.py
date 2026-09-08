"""Central configuration for SatQuery AI.

All tuneable constants, model IDs, thresholds, and file-format lists live here.
Nothing should be hardcoded elsewhere in the codebase that belongs in this file.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Device selection (lazy — avoids importing torch at module level for tests)
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Allow override via environment variable for CI / headless environments.
DEVICE: str = os.environ.get("SATQUERY_DEVICE", get_device())

# ---------------------------------------------------------------------------
# Model IDs (Hugging Face Hub)
# ---------------------------------------------------------------------------

PHRASING_MODEL_ID: str = os.environ.get(
    "SATQUERY_PHRASING_MODEL",
    "microsoft/Phi-3-mini-4k-instruct",
)

INTENT_PARSER_MODEL_ID: str = os.environ.get(
    "SATQUERY_INTENT_MODEL",
    "microsoft/Phi-3-mini-4k-instruct",
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

ROUTER_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("SATQUERY_ROUTER_THRESHOLD", "0.5")
)

# ---------------------------------------------------------------------------
# Modality detection thresholds
# ---------------------------------------------------------------------------

# SAR imagery is typically single-pol (1 band) or dual-pol (2 bands).
SAR_BAND_COUNT_MAX: int = 2

# Optical imagery has at least 3 bands (RGB or multispectral).
OPTICAL_BAND_COUNT_MIN: int = 3

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Spectral index thresholds (used by specialist modules)
# ---------------------------------------------------------------------------

NDVI_THRESHOLD: float = 0.3          # vegetation presence
NDWI_THRESHOLD: float = 0.2          # water body presence
BUILT_UP_THRESHOLD: float = 0.25     # built-up area index


@dataclass
class SpectralSettings:
    ndvi_sparse_vegetation: float = float(os.environ.get("SATQUERY_NDVI_THRESHOLD", "0.2"))
    ndwi_water_body: float = float(os.environ.get("SATQUERY_NDWI_THRESHOLD", "0.1"))
    ndbi_built_up: float = float(os.environ.get("SATQUERY_NDBI_THRESHOLD", "0.1"))


@dataclass
class SARSettings:
    vv_water_max_db: float = float(os.environ.get("SATQUERY_SAR_VV_WATER_MAX_DB", "-15.0"))
    vv_urban_min_db: float = float(os.environ.get("SATQUERY_SAR_VV_URBAN_MIN_DB", "-8.0"))
    vh_urban_min_db: float = float(os.environ.get("SATQUERY_SAR_VH_URBAN_MIN_DB", "-14.0"))




# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------

SUPPORTED_GEOTIFF_EXTENSIONS: frozenset[str] = frozenset({
    ".tif", ".tiff", ".geotiff",
})

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg",
})

ALL_SUPPORTED_EXTENSIONS: frozenset[str] = (
    SUPPORTED_GEOTIFF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
)

# ---------------------------------------------------------------------------
# Image processing defaults
# ---------------------------------------------------------------------------

MAX_IMAGE_DIM: int = 2048            # resize cap before feeding to models
DEFAULT_CRS: str = "EPSG:4326"

# ---------------------------------------------------------------------------
# Executor / specialist retry policy
# ---------------------------------------------------------------------------

SPECIALIST_MAX_RETRIES: int = 1      # retry once, then fail loudly

# ---------------------------------------------------------------------------
# Co-registration tolerance for image pairs
# ---------------------------------------------------------------------------

CRS_MISMATCH_TOLERANCE: float = 0.0          # exact CRS match required
BOUNDS_OVERLAP_TOLERANCE: float = 0.01        # fraction of extent allowed to differ


# ---------------------------------------------------------------------------
# Global settings instance
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    spectral: SpectralSettings = field(default_factory=SpectralSettings)
    sar: SARSettings = field(default_factory=SARSettings)
    device: str = DEVICE
    max_image_dim: int = MAX_IMAGE_DIM
    default_crs: str = DEFAULT_CRS


settings = Settings()

