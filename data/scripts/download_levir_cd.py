#!/usr/bin/env python3
"""
Download the LEVIR-CD dataset into data/raw/levir_cd/.

LEVIR-CD (Hao Chen et al., 2020): A large-scale building change detection
dataset with 637 bi-temporal Google Earth image pairs (1024x1024 pixels),
split into train/val/test sets with binary change masks.

The dataset is distributed as a zip archive (~600 MB) on Google Drive.
This script uses 'gdown' to handle Google Drive's confirmation pages.

Usage:
    python data/scripts/download_levir_cd.py
    python data/scripts/download_levir_cd.py --output-dir path/to/output
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


# Google Drive file IDs for LEVIR-CD
# The dataset is commonly hosted across several mirrors — we try them in order.
GDRIVE_FILE_IDS = [
    "1RIHK3wF4gRpKFajKhhrMmbVLVsuHxoQ_",  # Common public mirror
]

# Alternative: Hugging Face Hub mirror (if available)
HF_REPO_ID = "LEVIR-CD/LEVIR-CD"  # Placeholder — check if this exists

# Expected structure after extraction
EXPECTED_SUBDIRS = ["train", "val", "test"]
EXPECTED_INNER = ["A", "B", "label"]


def _download_via_gdown(file_id: str, output_path: Path) -> bool:
    """Download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print(
            "ERROR: 'gdown' is required for Google Drive downloads.\n"
            "Install it with: pip install gdown>=4.7.0",
            file=sys.stderr,
        )
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading LEVIR-CD from Google Drive (ID: {file_id})...")
    print(f"Target: {output_path}")

    try:
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        return output_path.exists() and output_path.stat().st_size > 1_000_000
    except Exception as e:
        print(f"gdown download failed: {e}", file=sys.stderr)
        return False


def _download_via_hf_hub(output_dir: Path) -> bool:
    """Try downloading from Hugging Face Hub mirror."""
    try:
        from huggingface_hub import snapshot_download

        print(f"Attempting download from Hugging Face Hub ({HF_REPO_ID})...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        return True
    except Exception as e:
        print(f"HF Hub download failed (may not exist as HF dataset): {e}")
        return False


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract the downloaded zip archive."""
    print(f"Extracting {zip_path.name} to {extract_to} ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_to))
    print(f"Extraction complete.")

    # Remove the zip to save space
    zip_path.unlink()
    print(f"Removed archive: {zip_path.name}")


def _validate_structure(dataset_dir: Path) -> bool:
    """Check that the extracted dataset has the expected structure."""
    # LEVIR-CD may extract into a subdirectory — find it
    candidates = [dataset_dir]
    for child in dataset_dir.iterdir():
        if child.is_dir():
            candidates.append(child)

    for candidate in candidates:
        found_splits = []
        for split in EXPECTED_SUBDIRS:
            split_dir = candidate / split
            if split_dir.is_dir():
                found_splits.append(split)
                for inner in EXPECTED_INNER:
                    inner_dir = split_dir / inner
                    if not inner_dir.is_dir():
                        continue

        if len(found_splits) >= 2:  # At least train + test
            print(f"\n[OK] Dataset validated at: {candidate}")
            print(f"  Splits found: {found_splits}")
            # Count images
            for split in found_splits:
                a_dir = candidate / split / "A"
                if a_dir.is_dir():
                    count = len(list(a_dir.glob("*.png"))) + len(list(a_dir.glob("*.jpg")))
                    print(f"  {split}/A: {count} images")
            return True

    print("WARNING: Could not validate expected LEVIR-CD structure.")
    print("Expected: train/{A,B,label}/, val/{A,B,label}/, test/{A,B,label}/")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download the LEVIR-CD change detection dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/raw/levir_cd/ relative to project root)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: project_root/data/raw/levir_cd/
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "raw" / "levir_cd"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"LEVIR-CD download target: {output_dir}")

    # Check if already downloaded
    if any(output_dir.iterdir()):
        print("Target directory is not empty. Checking existing content...")
        if _validate_structure(output_dir):
            print("Dataset already present and validated. Skipping download.")
            return
        print("Existing content does not match expected structure. Re-downloading...")

    # Strategy 1: gdown from Google Drive
    zip_path = output_dir / "LEVIR-CD.zip"
    downloaded = False

    for file_id in GDRIVE_FILE_IDS:
        if _download_via_gdown(file_id, zip_path):
            downloaded = True
            break

    # Strategy 2: Hugging Face Hub fallback
    if not downloaded:
        print("\nGoogle Drive download failed. Trying Hugging Face Hub...")
        if _download_via_hf_hub(output_dir):
            _validate_structure(output_dir)
            return

    if not downloaded:
        print(
            "\n" + "=" * 70 + "\n"
            "AUTOMATIC DOWNLOAD FAILED\n\n"
            "Please download LEVIR-CD manually:\n"
            "1. Visit: https://justchenhao.github.io/LEVIR/\n"
            "2. Download the dataset zip (~600 MB)\n"
            "3. Extract to: {}\n"
            "4. Expected structure: train/{A,B,label}/, val/, test/\n"
            .format(output_dir) +
            "=" * 70,
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract
    _extract_zip(zip_path, output_dir)

    # Validate
    _validate_structure(output_dir)

    print("\n[OK] LEVIR-CD download complete!")


if __name__ == "__main__":
    main()
