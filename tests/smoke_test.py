#!/usr/bin/env python3
"""
Smoke test for SatQuery AI specialist modules.

Loads real demo images and runs the actual model inference — NOT stubs.
Will download HF models on first run (~200 MB for CLIPSeg, ~15 MB for TinyCD).

Usage:
    python tests/smoke_test.py

Expected output:
    - Change detection result dict for a bitemporal pair
    - Grounding result dict for a single optical image with "highlight the water body"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Safe console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image


def _separator(title: str) -> None:
    """Print a visible section separator."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def _load_image(path: Path) -> np.ndarray:
    """Load an image file as a numpy array."""
    if not path.exists():
        print(f"  WARNING: Demo image not found: {path}")
        print(f"  Creating a synthetic placeholder image instead.")
        return np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    print(f"  Loaded: {path.name} ({arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]})")
    return arr


def test_change_detection() -> bool:
    """Test run_change_detection() on the demo bitemporal pair."""
    _separator("CHANGE DETECTION -- TinyCD")

    demo_dir = project_root / "data" / "demo_samples" / "bitemporal_pairs"

    before_path = demo_dir / "levir_sample_t1_before.png"
    after_path = demo_dir / "levir_sample_t2_after.png"

    print("Loading bitemporal pair:")
    before_img = _load_image(before_path)
    after_img = _load_image(after_path)

    print("\nRunning change detection...")
    t0 = time.time()

    try:
        from satquery.specialists.tinycd_change import run_change_detection

        result = run_change_detection(before_img, after_img)
        elapsed = time.time() - t0

        print(f"\n  [OK] Completed in {elapsed:.2f}s")
        print(f"\n  Result:")
        print(f"    answer:          {result['answer']}")
        print(f"    raw_confidence:  {result['raw_confidence']}")
        print(f"    evidence shape:  {result['evidence'].shape}")
        print(f"    evidence dtype:  {result['evidence'].dtype}")
        print(f"    change_summary:")
        for k, v in result["change_summary"].items():
            print(f"      {k}: {v}")

        # Basic sanity checks
        assert isinstance(result["answer"], str), "answer must be a string"
        assert isinstance(result["raw_confidence"], float), "raw_confidence must be a float"
        assert isinstance(result["evidence"], np.ndarray), "evidence must be ndarray"
        assert result["evidence"].dtype == bool, "evidence must be boolean mask"
        assert "pct_area_changed" in result["change_summary"], "change_summary must have pct_area_changed"
        assert isinstance(result["change_summary"]["class_before"], list), "class_before must be a list"
        assert isinstance(result["change_summary"]["class_after"], list), "class_after must be a list"

        print(f"\n  [OK] All assertions passed!")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] FAILED after {elapsed:.2f}s")
        print(f"    Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grounding() -> bool:
    """Test run_grounding() on the demo optical image with a water body query."""
    _separator("GROUNDING -- CLIPSeg")

    demo_dir = project_root / "data" / "demo_samples" / "single_optical"
    image_path = demo_dir / "sample_coastal_port.png"

    print("Loading single optical image:")
    image = _load_image(image_path)

    text_prompt = "highlight the water body"
    print(f"\nRunning grounding with prompt: \"{text_prompt}\"")
    t0 = time.time()

    try:
        from satquery.specialists.clipseg_grounding import run_grounding

        result = run_grounding(image, text_prompt)
        elapsed = time.time() - t0

        print(f"\n  [OK] Completed in {elapsed:.2f}s")
        print(f"\n  Result:")
        print(f"    answer:          {result['answer']}")
        print(f"    raw_confidence:  {result['raw_confidence']}")
        print(f"    evidence shape:  {result['evidence'].shape}")
        print(f"    evidence dtype:  {result['evidence'].dtype}")

        # Coverage stats
        total = result["evidence"].size
        positive = np.sum(result["evidence"])
        print(f"    mask coverage:   {positive}/{total} ({positive/total*100:.1f}%)")

        # Basic sanity checks
        assert isinstance(result["answer"], str), "answer must be a string"
        assert isinstance(result["raw_confidence"], float), "raw_confidence must be a float"
        assert isinstance(result["evidence"], np.ndarray), "evidence must be ndarray"
        assert result["evidence"].dtype == bool, "evidence must be boolean mask"

        # Quality warning check
        if result["raw_confidence"] < 0.35:
            print(f"\n  [WARN] QUALITY WARNING: Confidence is {result['raw_confidence']:.2f}")
            print(f"    CLIPSeg was trained on natural images, not satellite imagery.")
            print(f"    This result may be unreliable -- discuss scope-cut with the team.")

        print(f"\n  [OK] All assertions passed!")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL] FAILED after {elapsed:.2f}s")
        print(f"    Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overlay() -> bool:
    """Test draw_overlay() produces a valid PIL Image."""
    _separator("OVERLAY -- draw_overlay()")

    try:
        from satquery.utils.overlay import draw_overlay

        # Create a test image and mask
        base = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=bool)
        mask[30:80, 40:100] = True

        result = draw_overlay(
            base_image=base,
            mask=mask,
            bbox=(40, 30, 100, 80),
            label="Test Region",
        )

        assert isinstance(result, Image.Image), "draw_overlay must return PIL.Image"
        assert result.size == (128, 128), f"Output size mismatch: {result.size}"
        print(f"  [OK] draw_overlay() returned {result.size} PIL Image")

        # Save test output
        out_path = project_root / "tests" / "fixtures" / "overlay_test_output.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path))
        print(f"  [OK] Saved test overlay to: {out_path}")

        return True

    except Exception as e:
        print(f"  [FAIL] FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  SatQuery AI -- Smoke Test")
    print("  Running real model inference on demo images")
    print("  (Models will be downloaded on first run)")
    print("=" * 70)

    results = {}

    # Test 1: Change Detection
    results["change_detection"] = test_change_detection()

    # Test 2: Grounding
    results["grounding"] = test_grounding()

    # Test 3: Overlay
    results["overlay"] = test_overlay()

    # Summary
    _separator("SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All smoke tests passed! [OK]")
    else:
        print("  Some tests failed. See errors above. [FAIL]")
        sys.exit(1)


if __name__ == "__main__":
    main()
