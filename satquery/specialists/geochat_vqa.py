"""
GeoChat-7B VQA / Captioning specialist.

Loads ``MBZUAI/geochat-7B`` (LLaVA-1.5 architecture) in 4-bit via
``bitsandbytes`` with optional LoRA adapter from ``models/geochat/lora_adapter/``.

Public API
----------
- :func:`load_model` — explicit model loading (called lazily if needed).
- :func:`run_vqa` — visual question answering.
- :func:`run_caption` — image captioning.
- :func:`get_specialist` — factory returning this module or the BLIP-2 fallback.

Return shape (all functions): ``{"answer": str, "raw_confidence": float, "evidence": None}``

The output is structured factual data — **not** free-form prose — so the
downstream Phrasing LLM can convert it to natural language without hallucinating
numeric claims.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_LORA_ADAPTER_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "geochat" / "lora_adapter"

# Module-level singletons (lazy-loaded)
_model: Optional["LlavaForConditionalGeneration"] = None
_processor: Optional["LlavaProcessor"] = None
_device: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_id: str = _MODEL_ID,
    lora_adapter_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> None:
    """Load GeoChat-7B in 4-bit with optional LoRA adapter.

    Parameters
    ----------
    model_id : str
        Hugging Face Hub model ID.
    lora_adapter_dir : str | Path | None
        Directory containing a PEFT LoRA adapter.  Defaults to
        ``models/geochat/lora_adapter/``.  Skipped if the directory does
        not exist or is empty.
    device : str | None
        Target device (auto-detected when *None*).
    """
    global _model, _processor, _device

    from transformers import (
        BitsAndBytesConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
    )

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading %s in 4-bit NF4 quantisation …", model_id)
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    try:
        _processor = LlavaProcessor.from_pretrained(model_id)
    except Exception:
        _processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # --- LoRA adapter -------------------------------------------------------
    adapter_dir = Path(lora_adapter_dir) if lora_adapter_dir else _LORA_ADAPTER_DIR
    adapter_config = adapter_dir / "adapter_config.json"

    if adapter_dir.is_dir() and adapter_config.exists():
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_dir)
        _model = PeftModel.from_pretrained(_model, str(adapter_dir))
        try:
            _model = _model.merge_and_unload()
            logger.info("LoRA adapter merged successfully.")
        except Exception as e:
            logger.info("LoRA adapter attached (unmerged 4-bit inference): %s", e)
    else:
        logger.info("No LoRA adapter found at %s — running base model zero-shot.", adapter_dir)

    _model.eval()
    logger.info("GeoChat-7B ready on %s.", _device)


def _ensure_loaded() -> None:
    """Lazy-load the model on first inference call."""
    if _model is None:
        load_model()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """Normalise diverse image inputs to a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] in (1, 3, 4) and image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _compute_confidence(scores: tuple[torch.Tensor, ...]) -> float:
    """Estimate generation confidence from token-level log-probabilities.

    Computes the mean token probability across all generated tokens.

    .. note::

        This is an approximation — proper calibrated confidence would require
        a held-out calibration set.  See the TODO below.
    """
    # TODO(confidence): Replace this heuristic with a calibrated confidence
    # estimator (e.g., temperature scaling on a held-out set).  The current
    # approach returns mean(exp(log_softmax)) over generated tokens, which
    # is a reasonable proxy for generation certainty but is NOT calibrated
    # probability.  If `output_scores=True` is not supported or returns
    # empty, we fall back to a fixed 0.5 placeholder.
    if not scores:
        return 0.5  # placeholder — model did not return scores

    try:
        log_probs: list[float] = []
        for step_logits in scores:
            # step_logits: (batch, vocab)
            probs = torch.softmax(step_logits[0], dim=-1)
            top_prob = probs.max().item()
            log_probs.append(math.log(top_prob + 1e-12))
        mean_log_prob = sum(log_probs) / len(log_probs)
        return round(math.exp(mean_log_prob), 4)
    except Exception:
        return 0.5  # placeholder


def _generate(pil_image: Image.Image, prompt_text: str, max_new_tokens: int = 256) -> dict:
    """Shared generation routine for VQA and captioning."""
    _ensure_loaded()
    assert _model is not None and _processor is not None

    # LLaVA-1.5 prompt format
    full_prompt = f"<image>\nUSER: {prompt_text}\nASSISTANT:"

    inputs = _processor(text=full_prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
    answer = _processor.decode(generated_ids, skip_special_tokens=True).strip()

    raw_confidence = _compute_confidence(outputs.scores)

    return {"answer": answer, "raw_confidence": raw_confidence, "evidence": None}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_vqa(image: Union[Image.Image, np.ndarray], question: str) -> dict:
    """Answer a visual question about a remote-sensing image.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray
        Input satellite / aerial image.
    question : str
        Natural-language question about the image.

    Returns
    -------
    dict
        ``{"answer": str, "raw_confidence": float, "evidence": None}``
    """
    pil = _to_pil(image)
    return _generate(pil, question)


def run_caption(image: Union[Image.Image, np.ndarray]) -> dict:
    """Generate a caption for a remote-sensing image.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray
        Input satellite / aerial image.

    Returns
    -------
    dict
        ``{"answer": str, "raw_confidence": float, "evidence": None}``
    """
    pil = _to_pil(image)
    prompt = (
        "Describe the contents of this remote sensing image in detail. "
        "List the visible land cover types, notable structures, and "
        "approximate spatial layout."
    )
    return _generate(pil, prompt)


# ---------------------------------------------------------------------------
# Fallback factory — swap to BLIP-2 if GeoChat is too slow
# ---------------------------------------------------------------------------
_USE_FALLBACK: bool = False


def use_fallback(flag: bool = True) -> None:
    """Switch the specialist factory to the BLIP-2 fallback.

    Call ``use_fallback(True)`` after the latency benchmark decides that
    GeoChat-7B exceeds the 3-second-per-query threshold.
    """
    global _USE_FALLBACK
    _USE_FALLBACK = flag


def get_specialist():
    """Return the active VQA/caption module (this module or BLIP-2 fallback).

    Both modules expose identical function signatures:
    ``load_model()``, ``run_vqa(image, question)``, ``run_caption(image)``.
    """
    if _USE_FALLBACK:
        from satquery.specialists import blip2_fallback

        return blip2_fallback
    return __import__(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible class wrapper for pipeline/executor.py
# ---------------------------------------------------------------------------
class GeoChatSpecialist:
    """Backward-compatible wrapper around the module-level API.

    Existing code that instantiates ``GeoChatSpecialist()`` and calls
    ``.answer_query()`` / ``.generate_caption()`` will continue to work.
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or _MODEL_ID

    def answer_query(self, image_arr: np.ndarray, query: str) -> dict:
        result = run_vqa(image_arr, query)
        return {
            "answer": result["answer"],
            "confidence": result["raw_confidence"],
            "model": f"{self.model_id} (4-bit)",
        }

    def generate_caption(self, image_arr: np.ndarray) -> dict:
        result = run_caption(image_arr)
        return {
            "caption": result["answer"],
            "confidence": result["raw_confidence"],
            "model": self.model_id,
        }
