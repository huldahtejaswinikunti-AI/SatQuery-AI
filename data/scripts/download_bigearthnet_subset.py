"""
Downloads/prepares a small BigEarthNet subset for dev + training, using
torchgeo's `BigEarthNet` dataset class -- NOT the raw `GFM-Bench/BigEarthNet`
Hugging Face `datasets` repo.

Why torchgeo and not `datasets.load_dataset("GFM-Bench/BigEarthNet")`:
verified during PRD research (this week, not from memory) that the
GFM-Bench HF repo currently fails to load under recent `datasets` versions
-- it ships a custom loading script, and the `datasets` library error is
literally "Dataset scripts are no longer supported." torchgeo avoids this
because it downloads the official archive directly rather than going
through a HF `datasets` loading script, and it's actively maintained
(Microsoft). See: https://torchgeo.readthedocs.io/en/stable/api/datasets.html#bigearthnet

Honest caveat: `download=True` fetches the FULL official train/val/test
split archive -- there is no partial-download API, so this script can't
avoid pulling more than `n_patches` worth of raw data on first run. On
Colab/Kaggle's datacenter bandwidth this is fast (the files are HF-hosted);
on a slow home connection it may not be -- budget time for this on Day 1,
don't discover it's slow the night before the demo. After downloading, this
script materializes only the requested number of samples into a lightweight
.npz cache under <out_dir>/ so the rest of the pipeline iterates fast
without repeatedly touching torchgeo's full dataset object.

Requires `torchgeo` -- NOT yet in requirements.txt as of the initial repo
skeleton; add it (`pip install torchgeo`) before running this.

Usage:
    cd data/scripts
    python download_bigearthnet_subset.py --bands all --num-classes 19 \
        --split train --n-patches 3000 --out-dir ../processed/bigearthnet_subset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def main(bands: str, num_classes: int, split: str, n_patches: int, root: str, out_dir: str) -> None:
    try:
        from torchgeo.datasets import BigEarthNet
    except ImportError as e:
        raise SystemExit(
            "torchgeo is not installed. `pip install torchgeo` (and add it to "
            "requirements.txt) before running this script."
        ) from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading BigEarthNet (split={split!r}, bands={bands!r}, num_classes={num_classes}) "
        f"from root={root!r} -- download=True triggers a full-archive fetch on first run "
        f"if it isn't already present there. This can take a while; see module docstring."
    )
    dataset = BigEarthNet(root=root, split=split, bands=bands, num_classes=num_classes, download=True)

    # Resolve label names defensively -- `class_sets` is present in the
    # torchgeo versions checked during PRD research, but pin your torchgeo
    # version and confirm this attribute exists rather than trusting it
    # blindly across versions.
    class_names = getattr(dataset, "class_sets", {}).get(num_classes) if hasattr(dataset, "class_sets") else None

    n = min(n_patches, len(dataset))
    print(f"Dataset has {len(dataset)} total samples in this split; materializing {n} into {out_dir}")

    manifest_samples = []
    for i in range(n):
        sample = dataset[i]
        image = sample["image"].numpy()                     # (C, H, W)
        label_mask = sample["label"].numpy().astype(bool)    # (num_classes,)

        if class_names:
            labels = [class_names[j] for j, present in enumerate(label_mask) if present]
        else:
            # Fallback if `class_sets` isn't available in your torchgeo version --
            # you'll get indices instead of names. Resolve manually if needed.
            labels = label_mask.nonzero()[0].tolist()

        out_path = out / f"sample_{i:06d}.npz"
        np.savez_compressed(out_path, image=image, label_mask=label_mask)
        manifest_samples.append({"file": out_path.name, "labels": labels})

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} saved")

    manifest = {
        "bands": bands, "num_classes": num_classes, "split": split,
        "n_samples": n, "band_layout": "see perception/io.py module docstring",
        "samples": manifest_samples,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {n} samples + manifest.json written to {out_dir}")
    print("Next: perception/io.py's bigearthnet_sample_to_bands(image, bands_mode) "
          "turns each saved `image` array into named bands (red, nir, vv, ...).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", choices=["s1", "s2", "all"], default="all",
                         help="s2-only is enough for the land-cover classifier "
                              "(training/train_landcover_classifier.py) and is a smaller "
                              "download; use 'all' once you need paired SAR for the fusion "
                              "module (specialists/fusion/).")
    parser.add_argument("--num-classes", type=int, choices=[19, 43], default=19)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--n-patches", type=int, default=3000,
                         help="How many samples to materialize into the lightweight .npz "
                              "cache. The upstream download is the full split archive "
                              "regardless of this number -- see module docstring.")
    parser.add_argument("--root", type=str, default="../raw/bigearthnet",
                         help="Where torchgeo stores/looks for the full downloaded archive.")
    parser.add_argument("--out-dir", type=str, default="../processed/bigearthnet_subset",
                         help="Where the lightweight per-sample .npz cache + manifest.json go.")
    args = parser.parse_args()
    main(args.bands, args.num_classes, args.split, args.n_patches, args.root, args.out_dir)
