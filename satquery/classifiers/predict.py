from __future__ import annotations
from typing import Optional
import numpy as np
from satquery.classifiers.land_cover import LandCoverModel

class LandCoverPredictor:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.model = LandCoverModel(checkpoint_path)

    def predict(self, image_arr: np.ndarray, threshold: float = 0.35, top_k: int = 5) -> list[dict]:
        probs = self.model.predict_probabilities(image_arr)
        sorted_c = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cls_name, prob in sorted_c:
            if prob >= threshold or len(results) < 2:
                results.append({"class_name": cls_name, "probability": prob})
            if len(results) >= top_k:
                break
        return results
