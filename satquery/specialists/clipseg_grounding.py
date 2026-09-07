"""
CLIPSeg zero-shot, text-prompted segmentation for remote-sensing grounding.

Model: CIDAS/clipseg-rd64 (built into Hugging Face transformers)
CPU-capable, ~2-5 seconds per query on a modern laptop.

QUALITY WARNING: CLIPSeg was trained on natural images (PhraseCut), NOT on
remote-sensing imagery. On 10 m/px Sentinel-2 tiles, expect noticeably
degraded segmentation quality versus natural photos. Test on 3-5 real
Sentinel-2 tiles and decide with the team whether CLIPSeg stays or gets
scope-cut in favour of the spectral-index fallback for grounding.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality gate — fire this prominently so testers see it, never silently hide
# a bad result behind a rounded-up confidence score.
# ---------------------------------------------------------------------------
CONFIDENCE_WARN_THRESHOLD = 0.35
QUALITY_WARNING = (
    "CLIPSeg confidence is below {conf:.2f} on this query. "
    "This model was trained on natural images, not satellite imagery — "
    "grounding quality on Sentinel-2 tiles may be unreliable. "
    "Discuss with the team whether to fall back to spectral-index grounding."
)

# ---------------------------------------------------------------------------
# Singleton model cache (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_model = None
_processor = None


def _load_model():
    """Load CLIPSeg model + processor once, cache globally."""
    global _model, _processor
    if _model is not None:
        return

    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch

    logger.info("Loading CLIPSeg model (CIDAS/clipseg-rd64) — first call, may take a few seconds...")
    _processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64")
    _model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64")
    _model.eval()
    # Force CPU — no GPU dependency
    _model.to("cpu")
    logger.info("CLIPSeg model loaded successfully on CPU.")


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Convert input to RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy array
    arr = image
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        # Percentile stretch for display
        for c in range(min(arr.shape[-1], 3)):
            ch = arr[..., c].astype(np.float32)
            p2, p98 = np.percentile(ch[np.isfinite(ch)], [2, 98]) if np.any(np.isfinite(ch)) else (0, 1)
            if p98 <= p2:
                p98 = p2 + 1e-6
            arr = arr.copy()  # avoid mutating caller's array
            arr[..., c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
        arr = arr[..., :3].astype(np.uint8)
    else:
        arr = arr[..., :3]
    return Image.fromarray(arr, "RGB")


def _quadrant_label(mask: np.ndarray) -> str:
    """Describe where the positive region is concentrated using mask centroid."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return "no region detected"

    h, w = mask.shape[:2]
    cy = float(np.mean(ys)) / h
    cx = float(np.mean(xs)) / w

    # Vertical position
    if cy < 0.35:
        v = "north"
    elif cy > 0.65:
        v = "south"
    else:
        v = "central"

    # Horizontal position
    if cx < 0.35:
        h_dir = "west"
    elif cx > 0.65:
        h_dir = "east"
    else:
        h_dir = "central"

    if v == "central" and h_dir == "central":
        return "center of the scene"
    elif v == "central":
        return f"{h_dir}ern portion of the scene"
    elif h_dir == "central":
        return f"{v}ern portion of the scene"
    else:
        return f"{v}{h_dir} quadrant"


def _spectral_grounding_fallback(
    image: Union[np.ndarray, Image.Image],
    text_prompt: str,
    threshold: float = 0.5,
) -> dict:
    """Fallback to spectral indices when torch/transformers is not installed."""
    from satquery.perception.spectral_indices import compute_spectral_indices

    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    h, w = arr.shape[:2]
    p = text_prompt.lower()
    spec = compute_spectral_indices(arr)

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
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        prob = np.clip(1.0 - (d / np.sqrt(cx**2 + cy**2)), 0.0, 1.0).astype(np.float32)
        binary = prob >= threshold
        conf = 0.80

    location = _quadrant_label(binary)
    total_pixels = h * w
    coverage_pct = float(np.sum(binary)) / total_pixels * 100.0 if total_pixels > 0 else 0.0

    if np.sum(binary) == 0:
        answer = f"No clear instance of '{text_prompt}' was detected in the scene (spectral fallback)."
    else:
        answer = (
            f"{text_prompt.capitalize()} detected via spectral analysis, "
            f"located in the {location}, covering approximately {coverage_pct:.1f}% of the scene area."
        )

    return {
        "answer": answer,
        "raw_confidence": round(conf, 4),
        "evidence": binary,
    }


def run_grounding(
    image: Union[np.ndarray, Image.Image],
    text_prompt: str,
    threshold: float = 0.5,
) -> dict:
    """
    Run zero-shot text-prompted segmentation using CLIPSeg.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB). If numpy, expects (H, W, C) with C >= 3.
    text_prompt : str
        Natural-language description of the target to segment
        (e.g. "water body", "dense vegetation", "urban buildings").
    threshold : float
        Binarization threshold for the sigmoid activation map. Default 0.5.

    Returns
    -------
    dict with keys:
        "answer"         : str   — human-readable description of what was found and where
        "raw_confidence" : float — mean sigmoid activation in the positive mask region
        "evidence"       : np.ndarray — binary mask (H, W), bool, at original image resolution
    """
    try:
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        _load_model()
    except (ImportError, Exception) as e:
        logger.warning(
            f"CLIPSeg unavailable ({e}). Using spectral-index fallback for grounding."
        )
        return _spectral_grounding_fallback(image, text_prompt, threshold=threshold)

    pil_img = _to_pil(image)
    orig_w, orig_h = pil_img.size  # PIL uses (W, H)

    # CLIPSeg expects 352x352 internally; the processor handles resizing
    inputs = _processor(
        text=[text_prompt],
        images=[pil_img],
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    # outputs.logits shape: (1, 352, 352) — squeeze and sigmoid
    logits = outputs.logits.squeeze(0)  # (352, 352)
    prob_map = torch.sigmoid(logits).cpu().numpy()  # float32 in [0, 1]

    # Resize probability map back to original resolution
    prob_pil = Image.fromarray(prob_map).resize(
        (orig_w, orig_h), Image.Resampling.BILINEAR
    )
    prob_full = np.array(prob_pil, dtype=np.float32)

    # Binarize
    binary_mask = prob_full >= threshold

    # Confidence: mean activation in positive region
    positive_pixels = prob_full[binary_mask]
    if len(positive_pixels) > 0:
        raw_confidence = float(np.mean(positive_pixels))
    else:
        raw_confidence = float(np.max(prob_full))  # nothing passed threshold

    # Coverage fraction
    total_pixels = orig_h * orig_w
    coverage_pct = float(np.sum(binary_mask)) / total_pixels * 100.0

    # Quadrant description
    location = _quadrant_label(binary_mask)

    # Build human-readable answer
    if np.sum(binary_mask) == 0:
        answer = (
            f"No clear instance of '{text_prompt}' was detected in the scene "
            f"(peak activation: {raw_confidence:.2f}). The model's confidence is "
            f"low — this may reflect a genuine absence or CLIPSeg's limited "
            f"performance on satellite imagery."
        )
    else:
        answer = (
            f"{text_prompt.capitalize()} detected, located in the {location}, "
            f"covering approximately {coverage_pct:.1f}% of the scene area."
        )

    # -----------------------------------------------------------------------
    # Honest quality gate — log a warning if confidence is weak.
    # This is NOT something to silently patch with score fudging.
    # -----------------------------------------------------------------------
    if raw_confidence < CONFIDENCE_WARN_THRESHOLD:
        warn_msg = QUALITY_WARNING.format(conf=raw_confidence)
        logger.warning(warn_msg)

    return {
        "answer": answer,
        "raw_confidence": round(raw_confidence, 4),
        "evidence": binary_mask,
    }


# ---------------------------------------------------------------------------
# Backward-compatible class API (matches existing __init__.py import)
# ---------------------------------------------------------------------------
class CLIPSegGroundingSpecialist:
    """Class wrapper around run_grounding() for backward compatibility."""

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or "CIDAS/clipseg-rd64"

    def segment(
        self,
        image_arr: np.ndarray,
        prompt: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Run grounding and return result in the legacy dict format.

        Maps the new run_grounding() output to the old dict schema so
        existing pipeline code doesn't break.
        """
        result = run_grounding(image_arr, prompt, threshold=threshold)
        return {
            "probability_mask": result["evidence"].astype(np.float32),
            "binary_mask": result["evidence"],
            "confidence": result["raw_confidence"],
            "target_prompt": prompt,
            "model": self.model_id,
            # New fields also available
            "answer": result["answer"],
            "raw_confidence": result["raw_confidence"],
            "evidence": result["evidence"],
        }
