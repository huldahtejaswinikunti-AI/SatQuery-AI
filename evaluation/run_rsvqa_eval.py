"""
RSVQA evaluation script — measures VQA accuracy on a held-out split.

Usage::

    python evaluation/run_rsvqa_eval.py \\
        --checkpoint models/geochat/lora_adapter \\
        --split held_out \\
        --data-dir data/raw/rsvqaxben \\
        --output evaluation/results/rsvqa_run.json
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

from evaluation.eval_utils import compute_token_f1, normalize_text


def run_eval(
    checkpoint: str = "models/geochat/lora_adapter",
    split: str = "held_out",
    data_dir: str = "data/raw/rsvqaxben",
    output: str = "evaluation/results/rsvqa_run.json",
    max_samples: int = 100,
) -> dict:
    """Evaluate GeoChat VQA on a held-out split.

    Parameters
    ----------
    checkpoint : str
        Path to the LoRA adapter directory (or base model ID).
    split : str
        Which split to evaluate (``"held_out"`` or ``"train"``).
    data_dir : str
        Root of the prepared RSVQAxBEN data.
    output : str
        Path to write the JSON results.
    max_samples : int
        Cap the number of evaluation samples for speed.

    Returns
    -------
    dict
        Evaluation results with accuracy, mean F1, and per-sample details.
    """
    from satquery.specialists.geochat_vqa import load_model, run_vqa

    # --- Load data ----------------------------------------------------------
    data_path = Path(data_dir) / f"{split}.json"
    if not data_path.exists():
        print(f"[run_rsvqa_eval] WARNING: Data file {data_path} not found.")
        print("  Falling back to a minimal synthetic evaluation.")
        samples = _synthetic_samples()
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    samples = samples[:max_samples]

    # --- Load model ---------------------------------------------------------
    adapter_dir = Path(checkpoint)
    if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
        print(f"[run_rsvqa_eval] Loading model with LoRA from {checkpoint}")
        load_model(lora_adapter_dir=checkpoint)
    else:
        print("[run_rsvqa_eval] No LoRA adapter found -- evaluating base model.")
        load_model()

    # --- Evaluate -----------------------------------------------------------
    exact_matches = 0
    f1_scores: list[float] = []
    details: list[dict] = []

    for i, sample in enumerate(samples):
        question = _extract_question(sample)
        gt_answer = _extract_answer(sample)

        # Create a dummy image if no real image is available
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        image_path = sample.get("image", "")
        if image_path and Path(image_path).exists():
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)
        else:
            img = dummy_img

        try:
            result = run_vqa(img, question)
            pred = result["answer"]
        except Exception as e:
            pred = f"[ERROR] {e}"

        # Scoring
        em = 1 if normalize_text(pred) == normalize_text(gt_answer) else 0
        f1 = compute_token_f1(pred, gt_answer)
        exact_matches += em
        f1_scores.append(f1)

        details.append({
            "id": sample.get("id", i),
            "question": question,
            "ground_truth": gt_answer,
            "prediction": pred,
            "exact_match": em,
            "token_f1": round(f1, 4),
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(samples)} evaluated")

    # --- Aggregate metrics --------------------------------------------------
    n = len(samples)
    accuracy = exact_matches / n if n > 0 else 0.0
    mean_f1 = sum(f1_scores) / n if n > 0 else 0.0

    results = {
        "split": split,
        "checkpoint": checkpoint,
        "num_samples": n,
        "accuracy": round(accuracy, 4),
        "mean_token_f1": round(mean_f1, 4),
        "details": details,
    }

    # --- Write results ------------------------------------------------------
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[run_rsvqa_eval] Done.")
    print(f"  Accuracy: {accuracy:.4f}  Mean Token F1: {mean_f1:.4f}")
    print(f"  Results -> {out_path}")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_question(sample: dict) -> str:
    convs = sample.get("conversations", [])
    if convs:
        text = convs[0].get("value", "")
        return text.replace("<image>\n", "").strip()
    return sample.get("question", "Describe this image.")


def _extract_answer(sample: dict) -> str:
    convs = sample.get("conversations", [])
    if len(convs) >= 2:
        return convs[1].get("value", "")
    return sample.get("answer", "")


def _synthetic_samples() -> list[dict]:
    """Minimal synthetic VQA pairs for testing the eval pipeline."""
    return [
        {
            "id": "synth_0",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat land cover is visible?"},
                {"from": "gpt", "value": "Agricultural land with sparse vegetation."},
            ],
        },
        {
            "id": "synth_1",
            "conversations": [
                {"from": "human", "value": "<image>\nIs there water in this image?"},
                {"from": "gpt", "value": "No water bodies are visible."},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoChat VQA on RSVQA held-out split.")
    parser.add_argument("--checkpoint", type=str, default="models/geochat/lora_adapter")
    parser.add_argument("--split", type=str, default="held_out", choices=["held_out", "train"])
    parser.add_argument("--data-dir", type=str, default="data/raw/rsvqaxben")
    parser.add_argument("--output", type=str, default="evaluation/results/rsvqa_run.json")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        split=args.split,
        data_dir=args.data_dir,
        output=args.output,
        max_samples=args.max_samples,
    )
