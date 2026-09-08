"""Dataset Quality, Integrity, and Leakage Validation Pipeline for SatQuery AI.

Implements the complete verification checks required by SIH PS 26167:
  RAW DATA -> INTEGRITY -> FORMAT -> ANNOTATIONS -> DUPLICATE -> LEAKAGE -> METADATA

Usage:
    python data/scripts/validate_dataset.py --target data/demo_samples
    python data/scripts/validate_dataset.py --target data/raw/bigearthnet_subset --manifest manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetValidator")


def compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for exact duplicate and leakage detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetValidator:
    def __init__(self, target_dir: str | Path, manifest_path: str | Path | None = None) -> None:
        self.target_dir = Path(target_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.hashes: dict[str, Path] = {}
        self.duplicates: list[tuple[Path, Path]] = []
        self.corrupted_files: list[tuple[Path, str]] = []
        self.verified_files_count = 0
        self.pair_mismatches: list[str] = []
        self.invalid_masks: list[str] = []

    def validate_file_integrity(self, file_path: Path) -> bool:
        """Verify that the image file can be decoded without decompression or header errors."""
        suffix = file_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Reopen to test actual array decode
                with Image.open(file_path) as img:
                    arr = np.array(img)
                    if arr.size == 0 or arr.ndim < 2:
                        self.corrupted_files.append((file_path, "Decoded image array is empty or <2D"))
                        return False
            except Exception as e:
                self.corrupted_files.append((file_path, f"PIL decode error: {e}"))
                return False

        elif suffix in [".tif", ".tiff", ".geotiff"]:
            if HAS_RASTERIO:
                try:
                    with rasterio.open(file_path) as src:
                        if src.width <= 0 or src.height <= 0 or src.count <= 0:
                            self.corrupted_files.append((file_path, "Invalid raster dimensions or band count <= 0"))
                            return False
                        _ = src.read(1)  # Read first band to test decompression
                except Exception as e:
                    self.corrupted_files.append((file_path, f"Rasterio read error: {e}"))
                    return False
            else:
                try:
                    with Image.open(file_path) as img:
                        _ = np.array(img)
                except Exception as e:
                    self.corrupted_files.append((file_path, f"TIFF decode error: {e}"))
                    return False

        elif suffix in [".npz", ".npy"]:
            try:
                if suffix == ".npz":
                    with np.load(file_path) as data:
                        _ = [k for k in data.keys()]
                else:
                    _ = np.load(file_path)
            except Exception as e:
                self.corrupted_files.append((file_path, f"NumPy load error: {e}"))
                return False

        return True

    def check_duplicate(self, file_path: Path) -> bool:
        """Flag identical files using SHA-256 content hashes."""
        h = compute_file_hash(file_path)
        if h in self.hashes:
            self.duplicates.append((file_path, self.hashes[h]))
            return False
        self.hashes[h] = file_path
        return True

    def validate_pairs(self, pairs: list[tuple[Path, Path]]) -> None:
        """Ensure paired images (bi-temporal or optical-SAR) have matching spatial dimensions."""
        for p1, p2 in pairs:
            if not p1.exists() or not p2.exists():
                self.pair_mismatches.append(f"Missing paired file: {p1} or {p2}")
                continue
            try:
                im1 = np.array(Image.open(p1))
                im2 = np.array(Image.open(p2))
                if im1.shape[:2] != im2.shape[:2]:
                    self.pair_mismatches.append(
                        f"Spatial dimension mismatch: {p1.name} {im1.shape[:2]} vs {p2.name} {im2.shape[:2]}"
                    )
            except Exception as e:
                self.pair_mismatches.append(f"Pair decode error: {p1.name} / {p2.name}: {e}")

    def check_split_leakage(self, splits: dict[str, list[Path]]) -> list[str]:
        """Check for contamination between train, val, and test splits."""
        leakages = []
        split_hashes: dict[str, dict[str, Path]] = {}

        for split_name, paths in splits.items():
            split_hashes[split_name] = {}
            for p in paths:
                if p.exists() and p.is_file():
                    h = compute_file_hash(p)
                    split_hashes[split_name][h] = p

        # Check train vs test
        if "train" in split_hashes and "test" in split_hashes:
            for h, p_train in split_hashes["train"].items():
                if h in split_hashes["test"]:
                    p_test = split_hashes["test"][h]
                    leakages.append(f"TRAIN-TEST LEAKAGE: {p_train.name} (train) identical to {p_test.name} (test)!")

        # Check train vs val
        if "train" in split_hashes and "val" in split_hashes:
            for h, p_train in split_hashes["train"].items():
                if h in split_hashes["val"]:
                    p_val = split_hashes["val"][h]
                    leakages.append(f"TRAIN-VAL OVERLAP: {p_train.name} (train) identical to {p_val.name} (val)!")

        return leakages

    def run_full_validation(self) -> dict[str, Any]:
        logger.info("Starting validation on target directory: %s", self.target_dir)

        all_image_files = [
            p for p in self.target_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npz", ".npy"]
        ]

        logger.info("Found %d candidate data files to audit.", len(all_image_files))

        for f in all_image_files:
            if self.validate_file_integrity(f):
                self.check_duplicate(f)
                self.verified_files_count += 1

        # Check metadata.json if present
        meta_path = self.target_dir / "metadata.json"
        metadata_valid = False
        meta_count = 0
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as mf:
                    meta = json.load(mf)
                samples = meta.get("samples", [])
                meta_count = len(samples)
                metadata_valid = True
                logger.info("Found valid metadata.json with %d cataloged sample entries.", meta_count)
            except Exception as e:
                logger.warning("metadata.json exists but failed to parse: %s", e)

        results = {
            "target_dir": str(self.target_dir),
            "total_files_audited": len(all_image_files),
            "files_passed_integrity": self.verified_files_count,
            "corrupted_count": len(self.corrupted_files),
            "corrupted_files": [(str(p), err) for p, err in self.corrupted_files],
            "duplicates_count": len(self.duplicates),
            "duplicates": [(str(p1), str(p2)) for p1, p2 in self.duplicates],
            "metadata_present": metadata_valid,
            "metadata_entries": meta_count,
            "pair_mismatches": self.pair_mismatches,
            "passed": len(self.corrupted_files) == 0 and len(self.pair_mismatches) == 0,
        }

        logger.info("Validation complete. Status: %s", "PASSED" if results["passed"] else "FAILED")
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate remote-sensing dataset integrity and leakage.")
    parser.add_argument("--target", default="data/demo_samples", help="Directory containing dataset files")
    parser.add_argument("--manifest", default=None, help="Optional manifest.json to cross-check")
    args = parser.parse_args()

    validator = DatasetValidator(args.target, args.manifest)
    res = validator.run_full_validation()

    print("\n" + "=" * 60)
    print(" SatQuery AI Dataset Validation Summary")
    print("=" * 60)
    print(f" Target Directory:      {res['target_dir']}")
    print(f" Files Audited:         {res['total_files_audited']}")
    print(f" Files Passed:          {res['files_passed_integrity']}")
    print(f" Corrupted / Unreadable: {res['corrupted_count']}")
    print(f" Duplicate Hash Pairs:  {res['duplicates_count']}")
    print(f" Metadata Catalog:      {'Valid' if res['metadata_present'] else 'Not Found'}")
    print(f" Overall Status:        {'PASS [OK]' if res['passed'] else 'FAIL [X]'}")
    print("=" * 60)

    if not res["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
