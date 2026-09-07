#!/usr/bin/env python3
"""
CDVQA evaluation harness.

Runs change-VQA evaluation against the CDVQA dataset (Yuan et al., built on
the SECOND dataset: 2,968 bi-temporal pairs, ~122K QA pairs, 6 land-cover
classes).

Usage:
    python evaluation/run_cdvqa_eval.py --split held_out
    python evaluation/run_cdvqa_eval.py --split test --data-dir data/raw/cdvqa
    python evaluation/run_cdvqa_eval.py --split held_out --output-dir evaluation/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from evaluation.eval_utils import compute_token_f1, normalize_text


def _load_cdvqa_annotations(data_dir: Path, split: str) -> list[dict]:
    """
    Load CDVQA annotations for the given split.

    Expected data layout (standard CDVQA distribution):
        data_dir/
            images/
                <pair_id>/
                    A.png       (before)
                    B.png       (after)
            QA/
                <split>.json   (or annotations_<split>.json)

    Each annotation entry:
        {
            "image_id": str,
            "question": str,
            "answer": str,
            "question_type": str  (optional)
        }
    """
    qa_dir = data_dir / "QA"
    images_dir = data_dir / "images"

    # Find annotation file — try several naming conventions
    candidates = [
        qa_dir / f"{split}.json",
        qa_dir / f"annotations_{split}.json",
        qa_dir / f"cdvqa_{split}.json",
        data_dir / f"{split}.json",
        data_dir / "annotations.json",
    ]

    ann_file = None
    for c in candidates:
        if c.exists():
            ann_file = c
            break

    if ann_file is None:
        print(f"WARNING: No annotation file found for split '{split}' in {data_dir}")
        print(f"Searched: {[str(c) for c in candidates]}")
        print("Falling back to synthetic evaluation samples.")
        return _synthetic_samples()

    with open(ann_file, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        annotations = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ["annotations", "questions", "data", "samples"]:
            if key in data:
                annotations = data[key]
                break
        else:
            annotations = list(data.values()) if data else []

    # Resolve image paths
    resolved = []
    for ann in annotations:
        img_id = str(ann.get("image_id", ann.get("img_id", ann.get("id", ""))))

        # Try to find the image pair
        pair_dir = images_dir / img_id
        if not pair_dir.is_dir():
            # Try without leading zeros, etc.
            pair_dir = images_dir / img_id.lstrip("0")

        before_path = None
        after_path = None

        if pair_dir.is_dir():
            for name in ["A.png", "A.jpg", "T1.png", "before.png"]:
                p = pair_dir / name
                if p.exists():
                    before_path = p
                    break
            for name in ["B.png", "B.jpg", "T2.png", "after.png"]:
                p = pair_dir / name
                if p.exists():
                    after_path = p
                    break

        resolved.append({
            "image_id": img_id,
            "question": ann.get("question", ""),
            "answer": ann.get("answer", ""),
            "question_type": ann.get("question_type", ann.get("type", "unknown")),
            "before_path": str(before_path) if before_path else None,
            "after_path": str(after_path) if after_path else None,
        })

    print(f"Loaded {len(resolved)} QA pairs from {ann_file.name}")
    return resolved


def _synthetic_samples() -> list[dict]:
    """
    Generate synthetic evaluation samples when real CDVQA data isn't available.
    Uses the demo bitemporal pair images.
    """
    project_root = Path(__file__).resolve().parent.parent
    demo_dir = project_root / "data" / "demo_samples" / "bitemporal_pairs"

    before_path = demo_dir / "levir_sample_t1_before.png"
    after_path = demo_dir / "levir_sample_t2_after.png"

    samples = [
        {
            "image_id": "demo_01",
            "question": "What has changed in this area?",
            "answer": "Built-up area has increased.",
            "question_type": "change_description",
            "before_path": str(before_path) if before_path.exists() else None,
            "after_path": str(after_path) if after_path.exists() else None,
        },
        {
            "image_id": "demo_02",
            "question": "Is there any new construction?",
            "answer": "Yes, new buildings have appeared.",
            "question_type": "yes_no",
            "before_path": str(before_path) if before_path.exists() else None,
            "after_path": str(after_path) if after_path.exists() else None,
        },
        {
            "image_id": "demo_03",
            "question": "What percentage of the area changed?",
            "answer": "Approximately 15% of the area shows changes.",
            "question_type": "quantitative",
            "before_path": str(before_path) if before_path.exists() else None,
            "after_path": str(after_path) if after_path.exists() else None,
        },
    ]
    return samples


def _run_single_qa(sample: dict) -> dict:
    """Run the SatQuery pipeline on a single QA pair and compute metrics."""
    from satquery.specialists.tinycd_change import run_change_detection

    prediction = ""
    error = None
    latency = 0.0

    try:
        # Load images
        if sample["before_path"] and sample["after_path"]:
            before_img = np.array(Image.open(sample["before_path"]).convert("RGB"))
            after_img = np.array(Image.open(sample["after_path"]).convert("RGB"))
        else:
            # Fallback: synthetic images
            before_img = np.full((64, 64, 3), 80, dtype=np.uint8)
            after_img = np.full((64, 64, 3), 160, dtype=np.uint8)

        t0 = time.time()
        result = run_change_detection(before_img, after_img)
        latency = time.time() - t0

        prediction = result["answer"]

    except Exception as e:
        error = str(e)
        prediction = ""

    # Compute metrics
    gt = sample["answer"]
    token_f1 = compute_token_f1(prediction, gt)
    exact_match = 1.0 if normalize_text(prediction) == normalize_text(gt) else 0.0

    return {
        "image_id": sample["image_id"],
        "question": sample["question"],
        "question_type": sample["question_type"],
        "ground_truth": gt,
        "prediction": prediction,
        "token_f1": round(token_f1, 4),
        "exact_match": exact_match,
        "latency_s": round(latency, 3),
        "error": error,
    }


def run_eval(
    split: str = "held_out",
    data_dir: str | None = None,
    output_dir: str = "evaluation/results",
) -> dict:
    """
    Run CDVQA evaluation.

    Parameters
    ----------
    split : str
        Dataset split to evaluate on.
    data_dir : str or None
        Path to CDVQA dataset root. If None, uses data/raw/cdvqa/.
    output_dir : str
        Directory to write results JSON.

    Returns
    -------
    dict with evaluation summary.
    """
    project_root = Path(__file__).resolve().parent.parent

    if data_dir:
        data_path = Path(data_dir)
    else:
        data_path = project_root / "data" / "raw" / "cdvqa"

    print(f"=" * 60)
    print(f"CDVQA Evaluation -- split: {split}")
    print(f"Data directory: {data_path}")
    print(f"=" * 60)

    # Load annotations
    annotations = _load_cdvqa_annotations(data_path, split)

    if not annotations:
        print("ERROR: No evaluation samples found.", file=sys.stderr)
        sys.exit(1)

    # Run evaluation
    results = []
    total = len(annotations)

    for i, sample in enumerate(annotations):
        print(f"\r  [{i+1}/{total}] Evaluating {sample['image_id']}...", end="", flush=True)
        result = _run_single_qa(sample)
        results.append(result)

    print()  # newline after progress

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    valid_results = [r for r in results if r["error"] is None]
    error_count = len(results) - len(valid_results)

    if valid_results:
        avg_f1 = np.mean([r["token_f1"] for r in valid_results])
        avg_em = np.mean([r["exact_match"] for r in valid_results])
        avg_latency = np.mean([r["latency_s"] for r in valid_results])
    else:
        avg_f1 = avg_em = avg_latency = 0.0

    # Per-question-type breakdown
    type_metrics: dict[str, dict] = {}
    for r in valid_results:
        qt = r["question_type"]
        if qt not in type_metrics:
            type_metrics[qt] = {"f1_scores": [], "em_scores": [], "count": 0}
        type_metrics[qt]["f1_scores"].append(r["token_f1"])
        type_metrics[qt]["em_scores"].append(r["exact_match"])
        type_metrics[qt]["count"] += 1

    type_summary = {}
    for qt, data in type_metrics.items():
        type_summary[qt] = {
            "count": data["count"],
            "avg_token_f1": round(float(np.mean(data["f1_scores"])), 4),
            "avg_exact_match": round(float(np.mean(data["em_scores"])), 4),
        }

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    summary = {
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "successful_samples": len(valid_results),
        "error_count": error_count,
        "metrics": {
            "avg_token_f1": round(float(avg_f1), 4),
            "avg_exact_match": round(float(avg_em), 4),
            "avg_latency_s": round(float(avg_latency), 3),
        },
        "per_question_type": type_summary,
        "individual_results": results,
    }

    # Write results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_path / f"cdvqa_{split}_{timestamp_str}.json"

    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"CDVQA Evaluation Results -- {split}")
    print(f"{'=' * 60}")
    print(f"  Total samples:      {total}")
    print(f"  Successful:         {len(valid_results)}")
    print(f"  Errors:             {error_count}")
    print(f"  Avg Token F1:       {avg_f1:.4f}")
    print(f"  Avg Exact Match:    {avg_em:.4f}")
    print(f"  Avg Latency:        {avg_latency:.3f}s")

    if type_summary:
        print(f"\n  Per-question-type breakdown:")
        for qt, metrics in type_summary.items():
            print(
                f"    {qt:25s}  n={metrics['count']:4d}  "
                f"F1={metrics['avg_token_f1']:.4f}  "
                f"EM={metrics['avg_exact_match']:.4f}"
            )

    print(f"\n  Results written to: {out_file}")
    print(f"{'=' * 60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run change-VQA evaluation against the CDVQA dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="held_out",
        help="Dataset split to evaluate (default: held_out)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to CDVQA dataset root (default: data/raw/cdvqa/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory for results JSON (default: evaluation/results/)",
    )
    args = parser.parse_args()

    run_eval(
        split=args.split,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
