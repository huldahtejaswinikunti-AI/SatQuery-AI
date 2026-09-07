"""
BLIP-2 fallback VQA / captioning path.

Model: Salesforce/blip2-opt-2.7b (Hugging Face, free)
Lighter/faster fallback if GeoChat-7B proves too slow on free-tier GPU.

IMPORTANT: This module exposes exactly the same three function signatures
as the geochat_vqa module should expose:
    load_model(device="cpu")
    run_vqa(image, question: str) -> dict
    run_caption(image) -> dict

The executor can swap between GeoChat and BLIP-2 without any other code
changing — just change the import path.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model cache
# ---------------------------------------------------------------------------
_model = None
_processor = None
_device = "cpu"


def load_model(device: str = "cpu") -> None:
    """
    Load BLIP-2 OPT-2.7B model and processor, cache globally.

    Parameters
    ----------
    device : str
        Device to load the model on. Default "cpu".
        Use "cuda" if a GPU is available for faster inference.
    """
    global _model, _processor, _device

    if _model is not None:
        return

    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    _device = device
    model_id = "Salesforce/blip2-opt-2.7b"

    logger.info(f"Loading BLIP-2 OPT-2.7B ({model_id}) on {device} ...")

    _processor = Blip2Processor.from_pretrained(model_id)

    # Use float16 for memory efficiency (~6 GB instead of ~12 GB)
    dtype = torch.float16 if device != "cpu" else torch.float32
    _model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    _model.to(device)
    _model.eval()

    logger.info(f"BLIP-2 loaded successfully on {device} (dtype={dtype}).")


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Convert input to RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = image
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        # Percentile stretch
        out = np.zeros_like(arr, dtype=np.float32)
        for c in range(min(arr.shape[-1], 3)):
            ch = arr[..., c].astype(np.float32)
            valid = ch[np.isfinite(ch)]
            if len(valid) == 0:
                continue
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 <= p2:
                p98 = p2 + 1e-6
            out[..., c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
        arr = out[..., :3].astype(np.uint8)
    else:
        arr = arr[..., :3]
    return Image.fromarray(arr, "RGB")


def run_vqa(
    image: Union[np.ndarray, Image.Image],
    question: str,
) -> dict:
    """
    Visual Question Answering using BLIP-2.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB).
    question : str
        Natural-language question about the image.

    Returns
    -------
    dict with keys:
        "answer"         : str   — human-readable answer
        "raw_confidence" : float — proxy confidence (sequence probability)
        "model"          : str   — model identifier
    """
    import torch

    load_model(_device)

    pil_img = _to_pil(image)

    inputs = _processor(
        images=pil_img,
        text=question,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=5,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_text = _processor.decode(
        outputs.sequences[0], skip_special_tokens=True
    ).strip()

    # Compute a proxy confidence from the generation scores
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        # Log-probability of the sequence → convert to [0, 1]
        log_prob = float(outputs.sequences_scores[0])
        raw_confidence = min(1.0, max(0.0, np.exp(log_prob)))
    else:
        raw_confidence = 0.7  # fallback when scores unavailable

    return {
        "answer": generated_text,
        "raw_confidence": round(raw_confidence, 4),
        "model": "Salesforce/blip2-opt-2.7b",
    }


def run_caption(
    image: Union[np.ndarray, Image.Image],
) -> dict:
    """
    Image captioning using BLIP-2.

    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image (RGB).

    Returns
    -------
    dict with keys:
        "caption"        : str   — generated caption
        "raw_confidence" : float — proxy confidence
        "model"          : str   — model identifier
    """
    import torch

    load_model(_device)

    pil_img = _to_pil(image)

    inputs = _processor(
        images=pil_img,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=5,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    caption = _processor.decode(
        outputs.sequences[0], skip_special_tokens=True
    ).strip()

    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        log_prob = float(outputs.sequences_scores[0])
        raw_confidence = min(1.0, max(0.0, np.exp(log_prob)))
    else:
        raw_confidence = 0.75

    return {
        "caption": caption,
        "raw_confidence": round(raw_confidence, 4),
        "model": "Salesforce/blip2-opt-2.7b",
    }


# ---------------------------------------------------------------------------
# Backward-compatible class API (matches existing __init__.py import)
# ---------------------------------------------------------------------------
class BLIP2FallbackSpecialist:
    """
    Class wrapper for backward compatibility with existing pipeline code.

    Wraps the module-level load_model / run_vqa / run_caption functions
    into the answer_query / generate_caption class API that existing
    __init__.py and pipeline/executor.py expect.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or "Salesforce/blip2-opt-2.7b"

    def answer_query(self, image_arr: np.ndarray, query: str) -> dict:
        """VQA — matches GeoChatSpecialist.answer_query() signature."""
        result = run_vqa(image_arr, query)
        return {
            "answer": result["answer"],
            "confidence": result["raw_confidence"],
            "model": result["model"],
        }

    def generate_caption(self, image_arr: np.ndarray) -> dict:
        """Captioning — matches GeoChatSpecialist.generate_caption() signature."""
        result = run_caption(image_arr)
        return {
            "caption": result["caption"],
            "confidence": result["raw_confidence"],
            "model": result["model"],
        }
