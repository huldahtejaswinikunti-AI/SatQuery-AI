"""
Smoke test -- load one demo image and print run_vqa() + predict() output.

Usage::

    python scripts/smoke_test.py

This script is designed to run on CPU (for the land-cover classifier) and
will gracefully skip GeoChat VQA if no GPU / CUDA is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = PROJECT_ROOT / "data" / "demo_samples" / "single_optical"


def main() -> None:
    print("=" * 70)
    print("  SatQuery AI -- Smoke Test")
    print("=" * 70)

    # --- 1. Find a demo image -----------------------------------------------
    demo_images = sorted(DEMO_DIR.glob("*"))
    demo_images = [p for p in demo_images if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")]

    if not demo_images:
        print(f"\n[!] No demo images found in {DEMO_DIR}")
        print("    Run `python data/scripts/download_demo_samples.py` first,")
        print("    or place a .png/.jpg image in data/demo_samples/single_optical/")
        # Create a synthetic test image so the rest of the test can still run
        import numpy as np
        print("\n    -> Using synthetic 256x256 RGB test image instead.\n")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image_name = "synthetic_test_image"
    else:
        image_path = demo_images[0]
        print(f"\n[IMG] Demo image: {image_path.name}")
        from PIL import Image
        import numpy as np
        pil_img = Image.open(image_path).convert("RGB")
        image = np.array(pil_img)
        image_name = image_path.name

    print(f"      Shape: {image.shape}  Dtype: {image.dtype}\n")

    # --- 2. Land-cover predict() (CPU-safe) ---------------------------------
    print("-" * 70)
    print("  Land-Cover Classifier -- predict()")
    print("-" * 70)
    try:
        from satquery.classifiers.predict import predict

        result = predict(image)
        print(f"  Labels:     {result['labels']}")
        print(f"  Confidence: {result['confidence']}")
        print("  [OK] predict() succeeded.\n")
    except Exception as e:
        print(f"  [FAIL] predict() failed: {e}\n")

    # --- 3. GeoChat VQA (requires GPU) --------------------------------------
    print("-" * 70)
    print("  GeoChat-7B -- run_vqa()")
    print("-" * 70)
    try:
        import torch

        if not torch.cuda.is_available():
            print("  [SKIP] No CUDA GPU detected -- skipping GeoChat VQA.")
            print("         (GeoChat-7B requires a GPU with >=15 GB VRAM.)\n")
        else:
            from satquery.specialists.geochat_vqa import run_vqa

            question = "What type of land cover is visible in this image?"
            print(f"  Question: {question}")
            result = run_vqa(image, question)
            print(f"  Answer:         {result['answer']}")
            print(f"  Raw confidence: {result['raw_confidence']}")
            print(f"  Evidence:       {result['evidence']}")
            print("  [OK] run_vqa() succeeded.\n")
    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}")
        print("         Install with: pip install transformers peft bitsandbytes accelerate\n")
    except Exception as e:
        print(f"  [FAIL] run_vqa() failed: {e}\n")

    # --- 4. GeoChat Caption (requires GPU) ----------------------------------
    print("-" * 70)
    print("  GeoChat-7B -- run_caption()")
    print("-" * 70)
    try:
        import torch

        if not torch.cuda.is_available():
            print("  [SKIP] No CUDA GPU -- skipping captioning.\n")
        else:
            from satquery.specialists.geochat_vqa import run_caption

            result = run_caption(image)
            print(f"  Caption:        {result['answer']}")
            print(f"  Raw confidence: {result['raw_confidence']}")
            print("  [OK] run_caption() succeeded.\n")
    except Exception as e:
        print(f"  [FAIL] run_caption() failed: {e}\n")

    # --- Summary ------------------------------------------------------------
    print("=" * 70)
    print(f"  Smoke test complete for: {image_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
