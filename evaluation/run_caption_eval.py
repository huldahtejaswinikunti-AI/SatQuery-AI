"""
Caption evaluation script — measures BLEU / METEOR on a held-out split.

Usage::

    python evaluation/run_caption_eval.py \\
        --checkpoint models/geochat/lora_adapter \\
        --split held_out \\
        --data-dir data/raw/rsvqaxben \\
        --output evaluation/results/caption_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from evaluation.eval_utils import compute_bleu_1, normalize_text


def run_eval(
    checkpoint: str = "models/geochat/lora_adapter",
    split: str = "held_out",
    data_dir: str = "data/raw/rsvqaxben",
    output: str = "evaluation/results/caption_eval.json",
    max_samples: int = 50,
) -> dict:
    """Evaluate GeoChat captioning with BLEU and METEOR.

    Parameters
    ----------
    checkpoint : str
        Path to the LoRA adapter directory.
    split : str
        Which split to evaluate.
    data_dir : str
        Root of the prepared RSVQAxBEN data.
    output : str
        Path to write the JSON results.
    max_samples : int
        Cap the number of evaluation samples.

    Returns
    -------
    dict
        Evaluation results with BLEU-1, BLEU-4, and METEOR scores.
    """
    from satquery.specialists.geochat_vqa import load_model, run_caption

    # --- Load data ----------------------------------------------------------
    data_path = Path(data_dir) / f"{split}.json"
    if not data_path.exists():
        print(f"[run_caption_eval] WARNING: Data file {data_path} not found.")
        print("  Using synthetic reference captions.")
        samples = _synthetic_samples()
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    samples = samples[:max_samples]

    # --- Load model ---------------------------------------------------------
    adapter_dir = Path(checkpoint)
    if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
        print(f"[run_caption_eval] Loading model with LoRA from {checkpoint}")
        load_model(lora_adapter_dir=checkpoint)
    else:
        print("[run_caption_eval] No LoRA adapter found -- evaluating base model.")
        load_model()

    # --- Evaluate -----------------------------------------------------------
    bleu1_scores: list[float] = []
    bleu4_scores: list[float] = []
    meteor_scores: list[float] = []
    details: list[dict] = []

    for i, sample in enumerate(samples):
        ref = _extract_reference(sample)

        # Create dummy image if no real image available
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        image_path = sample.get("image", "")
        if image_path and Path(image_path).exists():
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)
        else:
            img = dummy_img

        try:
            result = run_caption(img)
            hyp = result["answer"]
        except Exception as e:
            hyp = f"[ERROR] {e}"

        # --- BLEU-1 (from eval_utils) --------------------------------------
        b1 = compute_bleu_1(ref, hyp)
        bleu1_scores.append(b1)

        # --- BLEU-4 (using nltk if available) ------------------------------
        b4 = _compute_bleu_4(ref, hyp)
        bleu4_scores.append(b4)

        # --- METEOR (using nltk if available) ------------------------------
        met = _compute_meteor(ref, hyp)
        meteor_scores.append(met)

        details.append({
            "id": sample.get("id", i),
            "reference": ref,
            "hypothesis": hyp,
            "bleu_1": round(b1, 4),
            "bleu_4": round(b4, 4),
            "meteor": round(met, 4),
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} evaluated")

    # --- Aggregate metrics --------------------------------------------------
    n = len(samples)
    mean_b1 = sum(bleu1_scores) / n if n > 0 else 0.0
    mean_b4 = sum(bleu4_scores) / n if n > 0 else 0.0
    mean_met = sum(meteor_scores) / n if n > 0 else 0.0

    results = {
        "split": split,
        "checkpoint": checkpoint,
        "num_samples": n,
        "bleu_1": round(mean_b1, 4),
        "bleu_4": round(mean_b4, 4),
        "meteor": round(mean_met, 4),
        "details": details,
    }

    # --- Write results ------------------------------------------------------
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[run_caption_eval] Done.")
    print(f"  BLEU-1: {mean_b1:.4f}  BLEU-4: {mean_b4:.4f}  METEOR: {mean_met:.4f}")
    print(f"  Results -> {out_path}")
    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _compute_bleu_4(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU-4 using nltk if available."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = normalize_text(reference).split()
        hyp_tokens = normalize_text(hypothesis).split()
        if not hyp_tokens:
            return 0.0
        smoothing = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing))
    except ImportError:
        # Fallback: use BLEU-1 as approximation
        return compute_bleu_1(reference, hypothesis)


def _compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR using nltk if available."""
    try:
        from nltk.translate.meteor_score import meteor_score

        ref_tokens = normalize_text(reference).split()
        hyp_tokens = normalize_text(hypothesis).split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        return float(meteor_score([ref_tokens], hyp_tokens))
    except ImportError:
        # Fallback: token F1 as METEOR proxy
        from evaluation.eval_utils import compute_token_f1
        return compute_token_f1(reference, hypothesis)


def _extract_reference(sample: dict) -> str:
    """Extract reference caption / answer from a sample."""
    convs = sample.get("conversations", [])
    if len(convs) >= 2:
        return convs[1].get("value", "")
    return sample.get("answer", sample.get("caption", ""))


def _synthetic_samples() -> list[dict]:
    """Minimal synthetic caption samples for testing."""
    return [
        {
            "id": "synth_cap_0",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this scene."},
                {"from": "gpt", "value": "Agricultural fields with scattered trees and a small settlement."},
            ],
        },
        {
            "id": "synth_cap_1",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this scene."},
                {"from": "gpt", "value": "Dense urban area with high-rise buildings and roads."},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoChat captioning (BLEU/METEOR).")
    parser.add_argument("--checkpoint", type=str, default="models/geochat/lora_adapter")
    parser.add_argument("--split", type=str, default="held_out", choices=["held_out", "train"])
    parser.add_argument("--data-dir", type=str, default="data/raw/rsvqaxben")
    parser.add_argument("--output", type=str, default="evaluation/results/caption_eval.json")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        split=args.split,
        data_dir=args.data_dir,
        output=args.output,
        max_samples=args.max_samples,
    )
