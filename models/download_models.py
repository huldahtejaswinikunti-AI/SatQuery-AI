"""
Model download registry for SatQuery AI.

Maps model names to Hugging Face Hub repo IDs and provides ``download_all()``
/ ``download_one()`` helpers.  Run as a script to download everything::

    python models/download_models.py              # download all
    python models/download_models.py --model geochat  # download one

Only ``geochat`` and ``land_cover`` are registered by this module.
Teammates: add your entries in the clearly marked section below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    # --- Core ML specialists (your entries) --------------------------------
    "geochat": "MBZUAI/geochat-7B",
    "land_cover": "satquery-ai/land-cover-resnet18-bigearthnet",
    # ^^ After fine-tuning, push the ResNet-18 checkpoint to this Hub repo
    # so teammates can pull it with `download_one("land_cover")`.

    # --- Vision & Change Specialists ---------------------------------------
    "clipseg": "CIDAS/clipseg-rd64",
    "tinycd": "AndreaCodegoni/Tiny_model_4_CD",
    # -----------------------------------------------------------------------
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_one(name: str, target_dir: str = "models") -> Path:
    """Download a single model by registry name.

    Parameters
    ----------
    name : str
        Key in :data:`MODEL_REGISTRY`.
    target_dir : str
        Root directory under which models are saved (each in a sub-folder).

    Returns
    -------
    Path
        The local directory where the model was downloaded.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub: pip install huggingface_hub"
        ) from exc

    repo_id = MODEL_REGISTRY[name]
    dest = Path(target_dir) / name
    print(f"Downloading {repo_id} → {dest} …")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(dest), resume_download=True)
        print(f"  ✓ {name} downloaded.")
    except Exception as e:
        print(f"  ✗ Failed to download {repo_id}: {e}")
    return dest


def download_all(target_dir: str = "models") -> None:
    """Download every model in :data:`MODEL_REGISTRY`."""
    for name in MODEL_REGISTRY:
        download_one(name, target_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SatQuery AI model weights from HF Hub.")
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Download a specific model. Choices: {', '.join(sorted(MODEL_REGISTRY.keys()))}",
    )
    parser.add_argument("--target-dir", type=str, default="models")
    args = parser.parse_args()

    if args.model:
        download_one(args.model, args.target_dir)
    else:
        download_all(args.target_dir)
