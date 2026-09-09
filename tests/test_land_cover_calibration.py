"""Regression tests for Land-Cover Classifier Calibration (Bug #1).

Verifies that:
1. When no checkpoint file is present, predict() explicitly flags `model_calibration: "untrained_fallback"`.
2. When the fine-tuned checkpoint exists, predict() sets `model_calibration: "calibrated"` and yields
   a top-1 prediction with a meaningful margin over top-2/top-3 (not a tight, random-head cluster).
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from satquery.classifiers.predict import predict, get_calibration_status, _DEFAULT_CHECKPOINT


@pytest.fixture
def fixture_rgb_image():
    """Create a synthetic RGB test fixture image with high contrast."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Give green vegetation dominant appearance in central area
    arr[:, :, 1] = 200
    arr[:50, :, 0] = 120
    return Image.fromarray(arr)


def test_missing_checkpoint_sets_untrained_fallback(fixture_rgb_image, tmp_path):
    """Assert predict() sets model_calibration: 'untrained_fallback' when checkpoint is missing."""
    non_existent_ckpt = tmp_path / "non_existent_model.pt"
    assert not non_existent_ckpt.exists()

    status = get_calibration_status(non_existent_ckpt)
    assert status == "untrained_fallback"

    res = predict(fixture_rgb_image, checkpoint_path=non_existent_ckpt)
    assert "model_calibration" in res
    assert res["model_calibration"] == "untrained_fallback"


def test_calibrated_checkpoint_has_meaningful_margin(fixture_rgb_image):
    """Assert genuine trained checkpoint exists, sets 'calibrated', and has real separation."""
    if not _DEFAULT_CHECKPOINT.exists():
        pytest.skip(f"Checkpoint {_DEFAULT_CHECKPOINT} not present on this runner.")

    status = get_calibration_status(_DEFAULT_CHECKPOINT)
    assert status == "calibrated"

    # Use a fixture image from demo_samples
    sample_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "demo_samples"
        / "single_optical"
        / "s2_coastal_harbor_rotterdam.png"
    )
    if sample_path.exists():
        img = Image.open(sample_path)
    else:
        img = fixture_rgb_image

    res = predict(img, checkpoint_path=_DEFAULT_CHECKPOINT)
    assert res["model_calibration"] == "calibrated"
    assert "top_k" in res
    assert len(res["top_k"]) >= 3

    top1 = res["top_k"][0]["probability"]
    top2 = res["top_k"][1]["probability"]
    margin = top1 - top2

    # A random head typically clusters within 0.05; a trained model separates dominant classes
    assert margin >= 0.05, f"Expected top-1 minus top-2 margin >= 0.05, got {margin:.4f} (top1={top1}, top2={top2})"
