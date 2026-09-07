"""
Download and prepare RSVQAxBEN data for GeoChat-7B LoRA fine-tuning.

RSVQAxBEN (Lobry et al., "RSVQA Meets BigEarthNet") provides VQA pairs
built on BigEarthNet Sentinel-2 patches.  This script fetches the dataset
and converts it to the LLaVA-style conversation format that the LoRA
fine-tuning notebook expects.

Output structure under ``data/raw/rsvqaxben/``::

    train.json        – LLaVA conversation list (training split)
    held_out.json     – LLaVA conversation list (eval split)
    images/           – symlink or copy of relevant BigEarthNet patches

Usage::

    python data/scripts/prepare_rsvqaxben.py
    python data/scripts/prepare_rsvqaxben.py --max-samples 5000 --held-out-frac 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(
    out_dir: str = "data/raw/rsvqaxben",
    max_samples: int = 5000,
    held_out_frac: float = 0.1,
    hf_dataset: str = "MBZUAI/GeoChat_Instruct",
) -> None:
    """Download RSVQAxBEN / GeoChat_Instruct and convert to LLaVA format.

    Parameters
    ----------
    out_dir : str
        Where to write the prepared JSON files.
    max_samples : int
        Maximum number of VQA pairs to include (cost / time tradeoff).
    held_out_frac : float
        Fraction of samples reserved for held-out evaluation.
    hf_dataset : str
        Hugging Face dataset ID.  Falls back to ``MBZUAI/GeoChat_Instruct``
        which includes RSVQAxBEN-derived examples.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install the `datasets` library: pip install datasets"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_rsvqaxben] Loading {hf_dataset} from Hugging Face ...")
    try:
        ds = load_dataset(hf_dataset, split="train")
    except Exception as e:
        print(f"  WARNING: Failed to load {hf_dataset}: {e}")
        print("  Falling back to synthetic placeholder data for dev/testing.")
        _write_placeholder(out, max_samples, held_out_frac)
        return

    # --- Filter to VQA-like examples and cap at max_samples -----------------
    # GeoChat_Instruct has a "conversations" field (list of dicts).
    # We filter for entries that look like single-turn VQA.
    samples: list[dict] = []
    for item in ds:
        convs = item.get("conversations", [])
        if len(convs) >= 2:
            question = convs[0].get("value", "")
            answer = convs[1].get("value", "")
            if question and answer:
                samples.append({
                    "id": item.get("id", f"sample_{len(samples)}"),
                    "image": item.get("image", ""),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": answer},
                    ],
                })
        if len(samples) >= max_samples:
            break

    if not samples:
        print("  WARNING: No VQA samples found -- writing placeholder data.")
        _write_placeholder(out, max_samples, held_out_frac)
        return

    # --- Train / held-out split ---------------------------------------------
    n_held_out = max(1, int(len(samples) * held_out_frac))
    held_out = samples[:n_held_out]
    train = samples[n_held_out:]

    _write_json(out / "train.json", train)
    _write_json(out / "held_out.json", held_out)

    print(f"[prepare_rsvqaxben] Done. Train: {len(train)}, Held-out: {len(held_out)}")
    print(f"  Files written to {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_json(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  -> {path.name}: {len(data)} samples")


def _write_placeholder(out: Path, max_samples: int, held_out_frac: float) -> None:
    """Generate minimal synthetic VQA pairs for dev/testing."""
    questions = [
        ("What type of land cover is visible?", "The image shows agricultural land."),
        ("Is there water in the scene?", "No water bodies are visible."),
        ("Describe the vegetation.", "Sparse vegetation cover with scattered trees."),
        ("Are there urban structures?", "Yes, residential buildings are present."),
        ("What is the dominant land use?", "The area is primarily used for farming."),
    ]

    samples: list[dict] = []
    for i in range(min(max_samples, 50)):
        q, a = questions[i % len(questions)]
        samples.append({
            "id": f"placeholder_{i:04d}",
            "image": f"placeholder_{i:04d}.png",
            "conversations": [
                {"from": "human", "value": f"<image>\n{q}"},
                {"from": "gpt", "value": a},
            ],
        })

    n_held_out = max(1, int(len(samples) * held_out_frac))
    _write_json(out / "train.json", samples[n_held_out:])
    _write_json(out / "held_out.json", samples[:n_held_out])
    print("  WARNING: Placeholder data written -- replace with real RSVQAxBEN for training.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare RSVQAxBEN / GeoChat_Instruct data for LoRA fine-tuning.",
    )
    parser.add_argument("--out-dir", type=str, default="data/raw/rsvqaxben")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum VQA pairs to prepare (default: 5000).")
    parser.add_argument("--held-out-frac", type=float, default=0.1,
                        help="Fraction reserved for held-out eval (default: 0.1).")
    parser.add_argument("--hf-dataset", type=str, default="MBZUAI/GeoChat_Instruct",
                        help="HF dataset ID for the VQA data.")
    args = parser.parse_args()

    main(
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        held_out_frac=args.held_out_frac,
        hf_dataset=args.hf_dataset,
    )
