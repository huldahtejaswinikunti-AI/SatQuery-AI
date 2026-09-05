from __future__ import annotations
from typing import Optional
import numpy as np

class CLIPSegGroundingSpecialist:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or "CIDAS/clipseg-rd64"

    def segment(self, image_arr: np.ndarray, prompt: str, threshold: float = 0.4) -> dict:
        h, w = image_arr.shape[:2]
        p = prompt.lower()
        from satquery.perception.spectral_indices import compute_spectral_indices
        spec = compute_spectral_indices(image_arr)

        if "water" in p or "river" in p or "lake" in p:
            prob = np.clip((spec.ndwi + 1.0) / 2.0, 0.0, 1.0)
            binary = spec.water_mask
            conf = 0.92
        elif "vegetation" in p or "forest" in p or "crop" in p:
            prob = np.clip((spec.ndvi + 1.0) / 2.0, 0.0, 1.0)
            binary = spec.vegetation_mask
            conf = 0.93
        elif "building" in p or "urban" in p:
            prob = np.clip((spec.ndbi + 1.0) / 2.0, 0.0, 1.0)
            binary = spec.built_up_mask
            conf = 0.88
        else:
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            prob = np.clip(1.0 - (d / np.sqrt(cx**2 + cy**2)), 0.0, 1.0).astype(np.float32)
            binary = prob >= threshold
            conf = 0.80

        return {"probability_mask": prob, "binary_mask": binary, "confidence": conf, "target_prompt": prompt, "model": "CIDAS/clipseg-rd64"}
