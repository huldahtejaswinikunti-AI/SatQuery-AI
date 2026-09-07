"""
Download BigEarthNet-S2 subset for land-cover classifier training.

Thin convenience wrapper around :mod:`download_bigearthnet_subset`, which does
the real work via ``torchgeo``.  Use this script from the repo root::

    python data/scripts/download_bigearth.py
    python data/scripts/download_bigearth.py --n-patches 5000 --bands s2

See ``download_bigearthnet_subset.py`` for full documentation on what
gets downloaded and where.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(
    n_patches: int = 3000,
    bands: str = "all",
    num_classes: int = 19,
    split: str = "train",
    out_dir: str = "data/processed/bigearthnet_subset",
    root: str = "data/raw/bigearthnet",
) -> None:
    """Download and prepare a BigEarthNet subset.

    Parameters
    ----------
    n_patches : int
        Number of patches to materialise into the ``.npz`` cache.
    bands : str
        ``"s1"`` (SAR only), ``"s2"`` (optical only), or ``"all"`` (both).
    num_classes : int
        Label granularity — 19 (CORINE-derived) or 43 (full CLC).
    split : str
        Dataset split: ``"train"``, ``"val"``, or ``"test"``.
    out_dir : str
        Output directory for the lightweight ``.npz`` sample cache.
    root : str
        Where ``torchgeo`` stores / looks for the full downloaded archive.
    """
    subset_script = Path(__file__).resolve().parent / "download_bigearthnet_subset.py"

    if not subset_script.exists():
        raise FileNotFoundError(
            f"Expected {subset_script} -- the detailed download script is "
            "missing from data/scripts/."
        )

    cmd = [
        sys.executable, str(subset_script),
        "--bands", bands,
        "--num-classes", str(num_classes),
        "--split", split,
        "--n-patches", str(n_patches),
        "--root", root,
        "--out-dir", out_dir,
    ]

    print(f"[download_bigearth] Delegating to {subset_script.name} ...")
    print(f"  command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download BigEarthNet-S2 subset (wrapper around download_bigearthnet_subset.py).",
    )
    parser.add_argument("--n-patches", type=int, default=3000,
                        help="Number of samples to materialise (default: 3000).")
    parser.add_argument("--bands", choices=["s1", "s2", "all"], default="all")
    parser.add_argument("--num-classes", type=int, choices=[19, 43], default=19)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out-dir", type=str, default="data/processed/bigearthnet_subset")
    parser.add_argument("--root", type=str, default="data/raw/bigearthnet")
    args = parser.parse_args()

    main(
        n_patches=args.n_patches,
        bands=args.bands,
        num_classes=args.num_classes,
        split=args.split,
        out_dir=args.out_dir,
        root=args.root,
    )
