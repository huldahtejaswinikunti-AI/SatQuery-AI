import numpy as np
import pytest
from satquery.cross_verification.verifier import (
    verify,
    CrossVerifier,
    VerificationResult,
)
from satquery.perception.spectral_indices import SpectralIndicesResult
from satquery.fusion.optical_sar_fusion import fuse, OpticalSARFusionEngine, FusionResult


def test_verify_no_deterministic_signal():
    claim = {"answer": "There is a large airport runway."}
    res = verify(claim, None)
    assert res["confidence_tag"] == "lower_confidence"
    assert "no deterministic cross-check available for this claim." in res["reason"]
    assert res["agreed"] is False


def test_verify_agreement_water():
    claim = {"answer": "Water is detected in the central reservoir."}
    signal = {"water_fraction": 0.35}
    res = verify(claim, signal)
    assert res["confidence_tag"] == "high_cross_verified"
    assert res["agreed"] is True
    assert "35.0%" in res["reason"]


def test_verify_disagreement_water():
    claim = {"answer": "Yes, water body is clearly visible."}
    signal = {"water_fraction": 0.0}
    res = verify(claim, signal)
    assert res["confidence_tag"] == "lower_confidence"
    assert res["agreed"] is False
    assert "Disagreement" in res["reason"]
    assert "0.0% water" in res["reason"]


def test_verify_agreement_vegetation():
    claim = {"answer": "Dense forest and green vegetation cover the area."}
    signal = {"vegetation_fraction": 0.72}
    res = verify(claim, signal)
    assert res["confidence_tag"] == "high_cross_verified"
    assert res["agreed"] is True
    assert "72.0%" in res["reason"]


def test_verify_disagreement_vegetation():
    claim = {"answer": "No vegetation observed, only barren land."}
    signal = {"vegetation_fraction": 0.65}
    res = verify(claim, signal)
    assert res["confidence_tag"] == "lower_confidence"
    assert res["agreed"] is False
    assert "Disagreement" in res["reason"]


def test_verify_purely_deterministic_source():
    claim = {
        "answer": "NDVI analysis confirms dense vegetation.",
        "source": "deterministic",
    }
    signal = {"vegetation_fraction": 0.8}
    res = verify(claim, signal)
    assert res["confidence_tag"] == "high_rule_based"
    assert res["agreed"] is True


def test_strict_contract_allowed_tags_only():
    """
    Ensure the returned confidence_tag STRICTLY adheres to the 3-state contract
    expected by pipeline executor:
    'high_cross_verified', 'lower_confidence', 'high_rule_based'
    """
    valid_tags = {"high_cross_verified", "lower_confidence", "high_rule_based"}

    res1 = verify({"answer": "water", "source": "deterministic"}, None)
    res2 = verify({"answer": "river"}, None)
    res3 = verify({"answer": "river"}, {"water_fraction": 0.5})
    res4 = verify({"answer": "river"}, {"water_fraction": 0.0})

    for r in [res1, res2, res3, res4]:
        assert r["confidence_tag"] in valid_tags


def test_backward_compat_cross_verifier_class():
    v = CrossVerifier()
    spec = SpectralIndicesResult(
        ndvi=np.zeros((4, 4)),
        ndwi=np.ones((4, 4)),
        ndbi=np.zeros((4, 4)),
        vegetation_mask=np.zeros((4, 4), bool),
        water_mask=np.ones((4, 4), bool),
        built_up_mask=np.zeros((4, 4), bool),
        vegetation_fraction=0.0,
        water_fraction=1.0,
        built_up_fraction=0.0,
    )
    res = v.verify_vqa_claim("Yes, water is present.", 0.9, spec, "Is there water?")
    assert isinstance(res, VerificationResult)
    assert res.is_cross_verified
    assert res.confidence_tag == "high_cross_verified"


def test_fuse_clear_sky_vegetation():
    h, w = 8, 8
    optical_indices = {
        "ndvi": np.full((h, w), 0.7, dtype=np.float32),
        "ndwi": np.full((h, w), -0.4, dtype=np.float32),
        "ndbi": np.full((h, w), -0.3, dtype=np.float32),
    }
    sar_masks = {
        "water_mask": np.zeros((h, w), dtype=bool),
        "builtup_mask": np.zeros((h, w), dtype=bool),
    }
    cloud_mask = np.zeros((h, w), dtype=bool)

    res = fuse(optical_indices, sar_masks, cloud_mask)
    assert res["land_cover_call"] == "vegetation"
    assert res["cloud_fraction"] == 0.0
    assert res["modality_weights"]["optical"] >= 0.70
    assert "vegetation" in res["reason"].lower()


def test_fuse_cloud_covered_sar_override_builtup():
    h, w = 8, 8
    optical_indices = {
        "ndvi": np.full((h, w), 0.1, dtype=np.float32),
        "ndwi": np.full((h, w), 0.1, dtype=np.float32),
        "ndbi": np.full((h, w), 0.1, dtype=np.float32),
    }
    sar_masks = {
        "water_mask": np.zeros((h, w), dtype=bool),
        "builtup_mask": np.ones((h, w), dtype=bool),
    }
    cloud_mask = np.ones((h, w), dtype=bool)

    res = fuse(optical_indices, sar_masks, cloud_mask)
    assert res["land_cover_call"] == "built-up"
    assert res["cloud_fraction"] == 1.0
    assert res["modality_weights"]["sar"] > res["modality_weights"]["optical"]
    assert "cloud" in res["reason"].lower()
    assert "sar" in res["reason"].lower()
    assert "built-up" in res["reason"].lower()


def test_fuse_cloud_covered_sar_override_water():
    h, w = 8, 8
    optical_indices = {
        "ndvi": np.full((h, w), 0.0, dtype=np.float32),
        "ndwi": np.full((h, w), 0.0, dtype=np.float32),
        "ndbi": np.full((h, w), 0.0, dtype=np.float32),
    }
    sar_masks = {
        "water_mask": np.ones((h, w), dtype=bool),
        "builtup_mask": np.zeros((h, w), dtype=bool),
    }
    cloud_mask = np.ones((h, w), dtype=bool)

    res = fuse(optical_indices, sar_masks, cloud_mask)
    assert res["land_cover_call"] == "water"
    assert "water" in res["reason"].lower()
