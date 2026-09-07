"""
Land-cover prediction interface.

Loads a fine-tuned :class:`~satquery.classifiers.land_cover.LandCoverModel`
checkpoint and exposes a single ``predict()`` entry-point that the pipeline
executor and smoke test can call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from satquery.classifiers.land_cover import (
    BIGEARTHNET_19_CLASSES,
    LandCoverModel,
    get_transforms,
)

# ---------------------------------------------------------------------------
# Module-level singleton (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_model: Optional[LandCoverModel] = None
_device: Optional[str] = None

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "models" / "land_cover" / "best_model.pt"


def _ensure_model(checkpoint_path: Optional[Union[str, Path]] = None) -> LandCoverModel:
    """Load the model once and cache it."""
    global _model, _device

    if _model is not None:
        return _model

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT

    if ckpt.exists():
        print(f"[land_cover] Loading fine-tuned checkpoint from {ckpt}")
        _model = LandCoverModel.load_from_checkpoint(ckpt, device=_device)
    else:
        print("[land_cover] No checkpoint found -- using ImageNet-pretrained backbone (untrained head).")
        _model = LandCoverModel(pretrained_backbone=True)
        _model.to(_device)
        _model.eval()

    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(
    image: Union[Image.Image, np.ndarray],
    threshold: float = 0.5,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Run multi-label land-cover prediction on a single image.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray
        Input image (RGB).  Numpy arrays are converted to PIL internally.
    threshold : float
        Sigmoid threshold for positive labels (default 0.5).
    checkpoint_path : str | Path | None
        Override the default checkpoint location.

    Returns
    -------
    dict
        ``{"labels": list[str], "confidence": float}``
        where *confidence* is the mean sigmoid probability of the active labels
        (or 0.0 if no label exceeds the threshold).
    """
    model = _ensure_model(checkpoint_path)
    tfm = get_transforms(train=False)

    # --- Normalise input to PIL ------------------------------------------------
    if isinstance(image, np.ndarray):
        # Handle (H, W, C) uint8 or float arrays
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        # Handle single-channel or >3 channels — take first 3
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = tfm(image).unsqueeze(0).to(_device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)              # (1, 19)
        probs = torch.sigmoid(logits)[0]    # (19,)

    active_indices = (probs >= threshold).nonzero(as_tuple=True)[0].tolist()

    if active_indices:
        labels = [BIGEARTHNET_19_CLASSES[i] for i in active_indices]
        confidence = float(probs[active_indices].mean().item())
    else:
        # Fall back to top-2 predictions regardless of threshold
        top2 = probs.topk(2).indices.tolist()
        labels = [BIGEARTHNET_19_CLASSES[i] for i in top2]
        confidence = float(probs[top2].mean().item())

    return {"labels": labels, "confidence": round(confidence, 4)}


# ---------------------------------------------------------------------------
# Backward-compatible class wrapper (used by existing pipeline/executor.py)
# ---------------------------------------------------------------------------
class LandCoverPredictor:
    """Thin class wrapper around :func:`predict` for backward compatibility."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._ckpt = checkpoint_path

    def predict(
        self,
        image_arr: np.ndarray,
        threshold: float = 0.35,
        top_k: int = 5,
    ) -> list[dict]:
        """Legacy interface — returns list of ``{"class_name", "probability"}``."""
        model = _ensure_model(self._ckpt)
        tfm = get_transforms(train=False)

        if isinstance(image_arr, np.ndarray):
            if image_arr.dtype != np.uint8:
                img = (np.clip(image_arr, 0, 1) * 255).astype(np.uint8)
            else:
                img = image_arr
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] > 3:
                img = img[..., :3]
            pil_img = Image.fromarray(img)
        else:
            pil_img = image_arr

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        tensor = tfm(pil_img).unsqueeze(0).to(_device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0]

        sorted_indices = probs.argsort(descending=True).tolist()
        results: list[dict] = []
        for idx in sorted_indices:
            p = float(probs[idx].item())
            if p >= threshold or len(results) < 2:
                results.append({"class_name": BIGEARTHNET_19_CLASSES[idx], "probability": round(p, 4)})
            if len(results) >= top_k:
                break
        return results
