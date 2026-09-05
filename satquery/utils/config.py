"""
Central configuration for SatQuery AI.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class SpectralThresholds:
    ndvi_dense_vegetation: float = 0.5
    ndvi_sparse_vegetation: float = 0.2
    ndwi_water_body: float = 0.0
    ndbi_built_up: float = 0.0

@dataclass
class SARThresholds:
    vv_water_max_db: float = -15.0
    vh_water_max_db: float = -22.0
    vv_urban_min_db: float = -8.0
    vh_urban_min_db: float = -14.0

@dataclass
class ModelConfig:
    geochat_id: str = "MBZUAI/geochat-7B"
    clipseg_id: str = "CIDAS/clipseg-rd64"
    tinycd_id: str = "AndreaCodegoni/Tiny_model_4_CD"
    blip2_id: str = "Salesforce/blip2-opt-2.7b"
    phrasing_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    land_cover_classes: list[str] = field(
        default_factory=lambda: [
            "Urban fabric", "Industrial or commercial units", "Arable land",
            "Permanent crops", "Pastures", "Complex cultivation patterns",
            "Land principally occupied by agriculture", "Broad-leaved forest",
            "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation", "Sclerophyllous vegetation",
            "Transitional woodland, shrub", "Beaches, dunes, sands",
            "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"
        ]
    )

@dataclass
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "models")
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    use_mock_fallbacks: bool = True
    device: str = os.getenv("SATQUERY_DEVICE", "auto")
    debug: bool = os.getenv("SATQUERY_DEBUG", "true").lower() in ("true", "1", "yes")
    spectral: SpectralThresholds = field(default_factory=SpectralThresholds)
    sar: SARThresholds = field(default_factory=SARThresholds)
    models: ModelConfig = field(default_factory=ModelConfig)

settings = Settings()
