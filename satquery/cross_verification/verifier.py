from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from satquery.perception.spectral_indices import SpectralIndicesResult

@dataclass
class VerificationResult:
    is_cross_verified: bool
    confidence_score: float
    confidence_tag: str
    evidence_signals: dict[str, Any]
    explanation: str

class CrossVerifier:
    def verify_vqa_claim(self, vlm_answer: str, vlm_confidence: float, spectral: SpectralIndicesResult, query: str) -> VerificationResult:
        ans, q = vlm_answer.lower(), query.lower()
        evidence = {"vegetation_fraction": spectral.vegetation_fraction, "water_fraction": spectral.water_fraction, "built_up_fraction": spectral.built_up_fraction}

        if any(w in q for w in ["water", "river", "lake", "ocean"]):
            vlm_claims = any(w in ans for w in ["yes", "detected", "present", "observed", "cover"])
            det_present = spectral.water_fraction >= 0.02
            if vlm_claims == det_present:
                return VerificationResult(True, min(0.96, vlm_confidence + 0.05), "high_cross_verified", evidence, f"Cross-verified with NDWI (water fraction: {spectral.water_fraction*100:.1f}%).")
            else:
                return VerificationResult(False, 0.65, "lower_confidence_disagreement", evidence, f"Disagreement: VLM answer contradicts NDWI water fraction ({spectral.water_fraction*100:.1f}%).")

        if any(w in q for w in ["vegetation", "forest", "crop", "green"]):
            vlm_claims = any(w in ans for w in ["yes", "dense", "forest", "present"])
            det_present = spectral.vegetation_fraction >= 0.15
            if vlm_claims == det_present:
                return VerificationResult(True, min(0.95, vlm_confidence + 0.04), "high_cross_verified", evidence, f"Corroborated by NDVI index (vegetation fraction: {spectral.vegetation_fraction*100:.1f}%).")
            else:
                return VerificationResult(False, 0.68, "lower_confidence_disagreement", evidence, f"Disagreement: VLM answer contradicts NDVI ({spectral.vegetation_fraction*100:.1f}%).")

        return VerificationResult(False, round(vlm_confidence, 2), "moderate_unverified", evidence, "Open-ended query without single-index counter-signal.")

    def verify_grounding_mask(self, predicted_mask: np.ndarray, target_entity: str, spectral: SpectralIndicesResult) -> VerificationResult:
        t = target_entity.lower()
        if "water" in t or "river" in t:
            ref = spectral.water_mask; name = "NDWI"
        elif "vegetation" in t or "forest" in t:
            ref = spectral.vegetation_mask; name = "NDVI"
        elif "building" in t or "urban" in t:
            ref = spectral.built_up_mask; name = "NDBI"
        else:
            return VerificationResult(False, 0.82, "moderate_unverified", {}, f"Target '{target_entity}' has no direct spectral equivalent.")

        inter = np.logical_and(predicted_mask, ref)
        union = np.logical_or(predicted_mask, ref)
        iou = float(np.sum(inter)) / float(np.sum(union) + 1e-6)
        evidence = {"iou": round(iou, 3)}

        if iou >= 0.35:
            return VerificationResult(True, min(0.96, 0.8 + iou*0.2), "high_cross_verified", evidence, f"High spatial alignment with {name} (IoU={iou:.2f}).")
        else:
            return VerificationResult(False, 0.64, "lower_confidence_disagreement", evidence, f"Low overlap with {name} (IoU={iou:.2f}).")
