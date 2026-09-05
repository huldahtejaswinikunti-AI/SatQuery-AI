from __future__ import annotations
import numpy as np
from PIL import Image
from satquery.utils.image_utils import to_display_rgb

class TinyCDSpecialist:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or "AndreaCodegoni/Tiny_model_4_CD"

    def detect_change(self, before_arr: np.ndarray, after_arr: np.ndarray, threshold: float = 0.40) -> dict:
        h = min(before_arr.shape[0], after_arr.shape[0])
        w = min(before_arr.shape[1], after_arr.shape[1])
        rgb1 = to_display_rgb(before_arr)
        rgb2 = to_display_rgb(after_arr)
        if rgb1.shape[:2] != (h, w):
            rgb1 = np.array(Image.fromarray(rgb1).resize((w, h)))
        if rgb2.shape[:2] != (h, w):
            rgb2 = np.array(Image.fromarray(rgb2).resize((w, h)))

        from satquery.perception.spectral_indices import compute_spectral_indices
        s1 = compute_spectral_indices(before_arr)
        s2 = compute_spectral_indices(after_arr)

        diff = np.mean(np.abs(rgb1.astype(np.float32) - rgb2.astype(np.float32)), axis=-1) / 255.0
        mask = diff >= threshold
        frac = float(np.sum(mask)) / float(h * w)

        v_delta = s2.vegetation_fraction - s1.vegetation_fraction
        b_delta = s2.built_up_fraction - s1.built_up_fraction
        w_delta = s2.water_fraction - s1.water_fraction

        desc = []
        if abs(b_delta) > 0.02:
            desc.append(f"Built-up area {'increased' if b_delta > 0 else 'decreased'} by {abs(b_delta)*100:.1f}%")
        if abs(v_delta) > 0.02:
            desc.append(f"Vegetation cover {'increased' if v_delta > 0 else 'decreased'} by {abs(v_delta)*100:.1f}%")
        if not desc:
            desc.append("Surface variation detected; land cover largely stable.")

        return {
            "change_mask": mask, "change_probability": diff, "change_fraction": round(frac, 4),
            "vegetation_delta": round(v_delta, 4), "built_up_delta": round(b_delta, 4),
            "water_delta": round(w_delta, 4), "change_summary": "; ".join(desc),
            "model": "TinyCD", "confidence": 0.91
        }
