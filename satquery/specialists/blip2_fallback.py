"""
BLIP-2 fallback specialist — same interface as :mod:`geochat_vqa`.

This module is a drop-in replacement for GeoChat-7B when the latency
benchmark determines that GeoChat exceeds the 3-second-per-query threshold.
The teammate who owns BLIP-2 integration will complete the model-loading
logic; the interface contract and return shapes are locked here.

Public API (mirrors ``geochat_vqa.py`` exactly)
------------------------------------------------
- :func:`load_model`
- :func:`run_vqa`
- :func:`run_caption`
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Module-level singletons (lazy-loaded)
_model = None
_processor = None
_device: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_id: str = _MODEL_ID,
    device: Optional[str] = None,
) -> None:
    """Load BLIP-2 OPT-2.7B for VQA / captioning.

    Parameters
    ----------
    model_id : str
        Hugging Face Hub model ID.
    device : str | None
        Target device (auto-detected when *None*).

    .. note::

        TODO: Teammate completes BLIP-2 loading.  Current implementation
        uses a lightweight mock so the rest of the pipeline stays runnable
        without pulling the full BLIP-2 weights.
    """
    global _model, _processor, _device
    import torch

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # TODO(teammate): Replace mock with real BLIP-2 loading:
    #
    #   from transformers import Blip2Processor, Blip2ForConditionalGeneration
    #   _processor = Blip2Processor.from_pretrained(model_id)
    #   _model = Blip2ForConditionalGeneration.from_pretrained(
    #       model_id, torch_dtype=torch.float16, device_map="auto",
    #   )
    #   _model.eval()
    #
    # For now, set a sentinel so _ensure_loaded() doesn't re-enter.
    _model = "mock"
    _processor = "mock"
    logger.warning(
        "[blip2_fallback] Using MOCK implementation — teammate: "
        "replace with real Blip2ForConditionalGeneration loading."
    )


def _ensure_loaded() -> None:
    if _model is None:
        load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Normalise input to PIL RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Public API — identical signatures and return shapes to geochat_vqa.py
# ---------------------------------------------------------------------------
def run_vqa(image: Union[Image.Image, np.ndarray], question: str) -> dict:
    """Answer a visual question about a remote-sensing image.

    Returns
    -------
    dict
        ``{"answer": str, "raw_confidence": float, "evidence": None}``
    """
    _ensure_loaded()
    pil = _to_pil(image)

    # TODO(teammate): Replace with real BLIP-2 inference.
    # Inputs:  _processor(images=pil, text=question, return_tensors="pt")
    # Outputs: _model.generate(**inputs, max_new_tokens=64)
    answer = f"[BLIP-2 MOCK] Unable to answer: '{question}' — model not loaded."
    return {"answer": answer, "raw_confidence": 0.0, "evidence": None}


def run_caption(image: Union[Image.Image, np.ndarray]) -> dict:
    """Generate a caption for a remote-sensing image.

    Returns
    -------
    dict
        ``{"answer": str, "raw_confidence": float, "evidence": None}``
    """
    _ensure_loaded()
    pil = _to_pil(image)

    # TODO(teammate): Replace with real BLIP-2 captioning.
    answer = "[BLIP-2 MOCK] Caption not available — model not loaded."
    return {"answer": answer, "raw_confidence": 0.0, "evidence": None}


# ---------------------------------------------------------------------------
# Backward-compatible class wrapper
# ---------------------------------------------------------------------------
class BLIP2FallbackSpecialist:
    """Class wrapper for pipeline backward compatibility."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or _MODEL_ID

    def answer_query(self, image_arr: np.ndarray, query: str) -> dict:
        result = run_vqa(image_arr, query)
        return {
            "answer": result["answer"],
            "confidence": result["raw_confidence"],
            "model": "BLIP-2 Fallback",
        }

    def generate_caption(self, image_arr: np.ndarray) -> dict:
        result = run_caption(image_arr)
        return {
            "caption": result["answer"],
            "confidence": result["raw_confidence"],
            "model": "BLIP-2 Fallback",
        }
