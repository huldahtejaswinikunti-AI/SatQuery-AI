import numpy as np
import pytest
from satquery.perception.sar_backscatter import (
    compute_sar_masks,
    analyze_sar_backscatter,
    linear_to_db,
    SARBackscatterResult,
)


def test_linear_to_db():
    res = linear_to_db(np.array([1.0, 10.0, 0.01]))
    assert np.isclose(res[0], 0.0, atol=1e-3)
    assert np.isclose(res[1], 10.0, atol=1e-3)
    assert np.isclose(res[2], -20.0, atol=1e-3)


def test_compute_sar_masks_water():
    """
    Test that low VV backscatter (<= -15.0 dB) is correctly classified as water.
    Literature baseline: Twele et al. 2016 C-band water specular threshold.
    """
    h, w = 4, 4
    vv = np.full((h, w), -22.0, dtype=np.float32)  # Low specular return
    vh = np.full((h, w), -28.0, dtype=np.float32)

    masks = compute_sar_masks(vv, vh)
    assert np.all(masks["water_mask"] == True)
    assert np.all(masks["builtup_mask"] == False)


def test_compute_sar_masks_builtup_double_bounce():
    """
    Test that high VV (>= -8 dB) AND high VH (>= -14 dB) classifies as built-up.
    Literature baseline: Ban & Jacob 2013 double-bounce corner reflection.
    """
    h, w = 4, 4
    vv = np.full((h, w), -5.0, dtype=np.float32)   # Strong double bounce
    vh = np.full((h, w), -10.0, dtype=np.float32)  # Depolarization from vertical structures

    masks = compute_sar_masks(vv, vh)
    assert np.all(masks["water_mask"] == False)
    assert np.all(masks["builtup_mask"] == True)


def test_compute_sar_masks_high_vv_only_rejects_urban():
    """
    Critical discrimination test: High VV alone without high VH (e.g. rough bare rock/soil)
    must NOT be classified as built-up. Both polarizations are mandatory for double bounce.
    """
    h, w = 4, 4
    vv = np.full((h, w), -6.0, dtype=np.float32)   # High VV
    vh = np.full((h, w), -19.0, dtype=np.float32)  # Low VH (no urban cross-polarization)

    masks = compute_sar_masks(vv, vh)
    assert not np.any(masks["builtup_mask"])
    assert not np.any(masks["water_mask"])


def test_compute_sar_masks_mismatched_shapes_raises():
    with pytest.raises(ValueError, match="does not match VH shape"):
        compute_sar_masks(np.zeros((4, 4)), np.zeros((4, 5)))


def test_backward_compat_analyze_sar_backscatter():
    h, w = 8, 8
    vv = np.full((h, w), -20.0, dtype=np.float32)
    vh = np.full((h, w), -25.0, dtype=np.float32)
    sar = np.stack([vv, vh], axis=-1)
    res = analyze_sar_backscatter(sar, is_already_db=True)

    assert isinstance(res, SARBackscatterResult)
    assert res.sar_water_fraction == 1.0
    assert res.sar_built_up_fraction == 0.0
    assert res.mean_vv_db == -20.0
