from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np

BIGEARTHNET_19_CLASSES = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture", "Broad-leaved forest",
    "Coniferous forest", "Mixed forest", "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation", "Sclerophyllous vegetation",
    "Transitional woodland, shrub", "Beaches, dunes, sands",
    "Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"
]

class LandCoverModel:
    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        self.classes = BIGEARTHNET_19_CLASSES
        self.num_classes = len(self.classes)

    def predict_probabilities(self, image_arr: np.ndarray) -> dict[str, float]:
        from satquery.perception.spectral_indices import compute_spectral_indices
        spec = compute_spectral_indices(image_arr)
        v, w, b = spec.vegetation_fraction, spec.water_fraction, spec.built_up_fraction

        scores = {}
        for c in self.classes:
            score = 0.05
            if "Urban" in c or "commercial" in c:
                score = min(0.95, b * 1.5 + 0.05)
            elif "forest" in c or "Pastures" in c or "vegetated" in c:
                score = min(0.95, v * 1.4 + 0.05)
            elif "water" in c or "wetlands" in c:
                score = min(0.98, w * 1.8 + 0.02)
            elif "Arable" in c or "cultivation" in c:
                score = min(0.85, (v * 0.7 + (1.0 - b - w) * 0.3))
            scores[c] = round(score, 4)
        return scores
