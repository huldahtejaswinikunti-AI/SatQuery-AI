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


def verify(vlm_claim: dict[str, Any], deterministic_signal: dict[str, Any] | None) -> dict[str, Any]:
    """
    Cross-verifies a Vision-Language Model (VLM) claim against physical, deterministic
    sensor observations (Sentinel-2 spectral indices and Sentinel-1 SAR backscatter).

    Contract Specification:
    -----------------------
    Output confidence_tag must be EXACTLY one of three valid string states:
    1. 'high_cross_verified': The VLM claim is corroborated by deterministic spectral/SAR evidence.
    2. 'lower_confidence': Either no deterministic check is available for the claim concept,
       OR deterministic sensor signals actively contradict the VLM claim.
    3. 'high_rule_based': The claim originated directly from a physical deterministic rule/index
       without VLM inference uncertainty.

    Args:
        vlm_claim: Output dictionary from specialist model or pipeline, containing at minimum:
            - "answer": str (the textual claim or hypothesis)
            - Optional "source": str (e.g. "deterministic", "geochat", "clipseg")
            - Optional "target": str (e.g. "water", "vegetation", "built-up")
        deterministic_signal: Matching spectral/SAR output dictionary for the same concept, or None.
            Can contain:
            - "water_fraction", "vegetation_fraction", "built_up_fraction" / "builtup_fraction"
            - "land_cover_call", "iou", "water_mask", "builtup_mask", etc.

    Returns:
        dict with:
            "confidence_tag": str ('high_cross_verified' | 'lower_confidence' | 'high_rule_based')
            "reason": str (human-readable explanation for SIH judges)
            "agreed": bool
            "details": dict of evidentiary signals
    """
    # 1. Check if the claim is purely deterministic (rule-based pipeline, no VLM inference)
    source = str(vlm_claim.get("source", "")).lower()
    if source in ("deterministic", "rule_based", "spectral_index", "spectral", "sar") or vlm_claim.get("is_deterministic") is True:
        return {
            "confidence_tag": "high_rule_based",
            "reason": "Claim derived directly from calibrated physical sensor measurements without model inference uncertainty.",
            "agreed": True,
            "details": {"source": source, "deterministic_signal": deterministic_signal},
        }

    # 2. Check if no deterministic cross-check is available
    if deterministic_signal is None or len(deterministic_signal) == 0:
        return {
            "confidence_tag": "lower_confidence",
            "reason": "no deterministic cross-check available for this claim.",
            "agreed": False,
            "details": {"cause": "missing_deterministic_signal"},
        }

    claim_text = str(vlm_claim.get("answer", "")).lower()
    target_entity = str(vlm_claim.get("target", "")).lower()

    # Spatial Grounding IoU Verification Pathway
    if "iou" in deterministic_signal:
        iou = float(deterministic_signal["iou"])
        iou_thresh = float(deterministic_signal.get("iou_threshold", 0.35))
        if iou >= iou_thresh:
            return {
                "confidence_tag": "high_cross_verified",
                "reason": f"VLM grounding segmentation spatially verified against deterministic physical mask (Intersection-over-Union: {iou:.2f} >= {iou_thresh}).",
                "agreed": True,
                "details": {"iou": round(iou, 3), "iou_threshold": iou_thresh},
            }
        else:
            return {
                "confidence_tag": "lower_confidence",
                "reason": f"Spatial disagreement: VLM localized region diverges from physical deterministic mask (IoU: {iou:.2f} below verification threshold {iou_thresh}).",
                "agreed": False,
                "details": {"iou": round(iou, 3), "iou_threshold": iou_thresh},
            }

    # Semantic Text Claim Verification Pathway
    def _is_claiming_presence(text: str) -> bool:
        negatives = ["no", "none", "not detected", "absent", "without", "zero", "false"]
        positives = ["yes", "detected", "present", "observed", "found", "there is", "shows", "cover", "dense", "water", "forest", "urban"]
        for neg in negatives:
            if neg in text:
                return False
        return any(pos in text for pos in positives)

    is_claiming_present = _is_claiming_presence(claim_text)

    # Concept: Water / Hydrology
    if any(k in claim_text or k in target_entity for k in ["water", "river", "lake", "ocean", "flood", "pond", "reservoir"]):
        water_frac = float(
            deterministic_signal.get("water_fraction")
            or deterministic_signal.get("water_mask_fraction")
            or deterministic_signal.get("fractions", {}).get("water", 0.0)
        )
        det_call = deterministic_signal.get("land_cover_call", "").lower()
        det_present = (water_frac >= 0.02) or (det_call == "water") or bool(deterministic_signal.get("water", False))

        if is_claiming_present == det_present:
            water_pct = round(water_frac * 100, 1)
            return {
                "confidence_tag": "high_cross_verified",
                "reason": f"Cross-verified with NDWI/SAR water masks: physical sensors confirm water presence status (measured water coverage: {water_pct}%).",
                "agreed": True,
                "details": {"concept": "water", "water_fraction": water_frac, "vlm_claims_present": is_claiming_present},
            }
        else:
            water_pct = round(water_frac * 100, 1)
            vlm_state = "present" if is_claiming_present else "absent"
            return {
                "confidence_tag": "lower_confidence",
                "reason": f"Disagreement: VLM claims water is {vlm_state}, but physical NDWI and SAR radar measure {water_pct}% water coverage.",
                "agreed": False,
                "details": {"concept": "water", "water_fraction": water_frac, "vlm_claims_present": is_claiming_present},
            }

    # Concept: Vegetation / Forestry
    if any(k in claim_text or k in target_entity for k in ["vegetation", "forest", "tree", "crop", "greenery", "agricultural"]):
        veg_frac = float(
            deterministic_signal.get("vegetation_fraction")
            or deterministic_signal.get("fractions", {}).get("vegetation", 0.0)
        )
        det_call = deterministic_signal.get("land_cover_call", "").lower()
        det_present = (veg_frac >= 0.15) or (det_call == "vegetation") or bool(deterministic_signal.get("vegetation", False))

        if is_claiming_present == det_present:
            veg_pct = round(veg_frac * 100, 1)
            return {
                "confidence_tag": "high_cross_verified",
                "reason": f"Corroborated by Sentinel-2 NDVI spectral reflectance (measured vegetation coverage: {veg_pct}%).",
                "agreed": True,
                "details": {"concept": "vegetation", "vegetation_fraction": veg_frac, "vlm_claims_present": is_claiming_present},
            }
        else:
            veg_pct = round(veg_frac * 100, 1)
            vlm_state = "present" if is_claiming_present else "absent"
            return {
                "confidence_tag": "lower_confidence",
                "reason": f"Disagreement: VLM claims vegetation is {vlm_state}, but physical NDVI reflectance indicates {veg_pct}% vegetation coverage.",
                "agreed": False,
                "details": {"concept": "vegetation", "vegetation_fraction": veg_frac, "vlm_claims_present": is_claiming_present},
            }

    # Concept: Built-up / Urban settlements
    if any(k in claim_text or k in target_entity for k in ["built-up", "building", "urban", "city", "structure", "infrastructure", "settlement"]):
        built_frac = float(
            deterministic_signal.get("built_up_fraction")
            or deterministic_signal.get("builtup_fraction")
            or deterministic_signal.get("fractions", {}).get("built_up", 0.0)
        )
        det_call = deterministic_signal.get("land_cover_call", "").lower()
        det_present = (built_frac >= 0.05) or (det_call == "built-up") or bool(deterministic_signal.get("built-up", False))

        if is_claiming_present == det_present:
            built_pct = round(built_frac * 100, 1)
            return {
                "confidence_tag": "high_cross_verified",
                "reason": f"Cross-verified by Sentinel-1 double-bounce radar backscatter and NDBI (built-up coverage: {built_pct}%).",
                "agreed": True,
                "details": {"concept": "built-up", "built_up_fraction": built_frac, "vlm_claims_present": is_claiming_present},
            }
        else:
            built_pct = round(built_frac * 100, 1)
            vlm_state = "present" if is_claiming_present else "absent"
            return {
                "confidence_tag": "lower_confidence",
                "reason": f"Disagreement: VLM claims built-up structures are {vlm_state}, but physical radar and NDBI measure {built_pct}% coverage.",
                "agreed": False,
                "details": {"concept": "built-up", "built_up_fraction": built_frac, "vlm_claims_present": is_claiming_present},
            }

    # General concept matching with land_cover_call
    if "land_cover_call" in deterministic_signal:
        det_call = deterministic_signal["land_cover_call"].lower()
        if det_call in claim_text:
            return {
                "confidence_tag": "high_cross_verified",
                "reason": f"Cross-verified: VLM claim matches deterministic fused land cover determination ('{det_call}').",
                "agreed": True,
                "details": {"land_cover_call": det_call},
            }
        else:
            return {
                "confidence_tag": "lower_confidence",
                "reason": f"Disagreement: VLM claim diverges from deterministic land cover call ('{det_call}').",
                "agreed": False,
                "details": {"land_cover_call": det_call},
            }

    # Fallback for claims with no matching deterministic parameter
    return {
        "confidence_tag": "lower_confidence",
        "reason": "no deterministic cross-check available for this claim.",
        "agreed": False,
        "details": {"cause": "unmatched_claim_concept"},
    }


class CrossVerifier:
    """
    Maintained for pipeline executor backward compatibility.
    """
    def verify_vqa_claim(self, vlm_answer: str, vlm_confidence: float, spectral: SpectralIndicesResult, query: str) -> VerificationResult:
        ans, q = vlm_answer.lower(), query.lower()
        evidence = {
            "vegetation_fraction": spectral.vegetation_fraction,
            "water_fraction": spectral.water_fraction,
            "built_up_fraction": spectral.built_up_fraction
        }

        # Use the verify() rule engine
        claim = {"answer": vlm_answer, "target": query}
        res = verify(claim, evidence)

        if res["confidence_tag"] == "high_cross_verified":
            return VerificationResult(True, min(0.96, vlm_confidence + 0.05), "high_cross_verified", evidence, res["reason"])
        else:
            return VerificationResult(False, 0.65, "lower_confidence_disagreement", evidence, res["reason"])

    def verify_grounding_mask(self, predicted_mask: np.ndarray, target_entity: str, spectral: SpectralIndicesResult) -> VerificationResult:
        t = target_entity.lower()
        if "water" in t or "river" in t:
            ref = spectral.water_mask
            name = "NDWI"
        elif "vegetation" in t or "forest" in t:
            ref = spectral.vegetation_mask
            name = "NDVI"
        elif "building" in t or "urban" in t:
            ref = spectral.built_up_mask
            name = "NDBI"
        else:
            return VerificationResult(False, 0.82, "moderate_unverified", {}, f"Target '{target_entity}' has no direct spectral equivalent.")

        inter = np.logical_and(predicted_mask, ref)
        union = np.logical_or(predicted_mask, ref)
        iou = float(np.sum(inter)) / float(np.sum(union) + 1e-6)

        claim = {"answer": f"Segmentation for {target_entity}", "target": target_entity}
        signal = {"iou": round(iou, 3)}
        res = verify(claim, signal)

        if res["confidence_tag"] == "high_cross_verified":
            return VerificationResult(True, min(0.96, 0.8 + iou*0.2), "high_cross_verified", {"iou": round(iou, 3)}, f"High spatial alignment with {name} (IoU={iou:.2f}).")
        else:
            return VerificationResult(False, 0.64, "lower_confidence_disagreement", {"iou": round(iou, 3)}, f"Low overlap with {name} (IoU={iou:.2f}).")
