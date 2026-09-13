"""Tests for Chandrayaan-2 Lunar Mode and Multi-Input Batch Processing."""
from __future__ import annotations

import numpy as np
import pytest

from satquery.lunar.lunar_validator import validate_lunar_input
from satquery.lunar.lunar_pipeline import run_lunar_pipeline
from app.pipeline_bridge import (
    run_lunar_pipeline as bridge_run_lunar,
    run_batch_pipeline,
    _BADGE_CONFIG,
)


def test_lunar_validator_valid():
    img = (np.random.rand(100, 100) * 255).astype(np.uint8)
    ok, err, meta = validate_lunar_input([img], [{}], "Identify craters")
    assert ok is True
    assert err is None
    assert meta["target_body"] == "Moon"
    assert meta["sensor"] == "Chandrayaan-2 OHRC / TMC-2"


def test_lunar_validator_empty_images():
    ok, err, _ = validate_lunar_input([], [], "Identify craters")
    assert ok is False
    assert "No lunar imagery" in err


def test_lunar_validator_multiple_images():
    img1 = (np.random.rand(50, 50) * 255).astype(np.uint8)
    img2 = (np.random.rand(50, 50) * 255).astype(np.uint8)
    ok, err, _ = validate_lunar_input([img1, img2], [{}, {}], "Identify craters")
    assert ok is False
    assert "one product at a time" in err


def test_lunar_validator_empty_query():
    img = (np.random.rand(50, 50) * 255).astype(np.uint8)
    ok, err, _ = validate_lunar_input([img], [{}], "   ")
    assert ok is False
    assert "Query cannot be empty" in err


def test_lunar_pipeline_contract():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    res = run_lunar_pipeline([img], [{"filename": "test_lunar.png"}], "Analyze crater rims and shadows")

    # Contract Assertions
    assert res["confidence_tag"] == "experimental_unverified"
    assert res["confidence"] == "experimental_unverified"
    assert res["confidence_score"] is None
    assert res["overlay"] is not None
    assert len(res["answer"]) > 20

    # Trace assertions
    trace = res["trace"]
    assert trace["task"] == "lunar_vqa_zero_shot"
    assert "cross_verification_skipped" in trace["tools_invoked"]
    assert "lunar" in trace["tools_invoked"][0]

    # Report markdown
    assert "Chandrayaan-2" in res["report_markdown"]
    assert "experimental_unverified" in res["report_markdown"]


def test_lunar_badge_config():
    assert "experimental_unverified" in _BADGE_CONFIG
    cfg = _BADGE_CONFIG["experimental_unverified"]
    assert "badge-experimental-unverified" in cfg["cls"]
    assert "Experimental" in cfg["label"]
    assert "No Cross-Check" in cfg["label"]


def test_batch_pipeline_sequential_and_error_isolation():
    img_valid = (np.random.rand(64, 64) * 255).astype(np.uint8)

    # Item 1: Valid
    it1 = {"images": [img_valid], "metas": [{"filename": "sample1.png"}], "query": "Describe craters"}
    # Item 2: Invalid (empty query)
    it2 = {"images": [img_valid], "metas": [{"filename": "sample2.png"}], "query": ""}
    # Item 3: Valid
    it3 = {"images": [img_valid], "metas": [{"filename": "sample3.png"}], "query": "Assess regolith"}

    batch = [it1, it2, it3]
    progress_calls = []

    def callback(idx, total, status, res):
        progress_calls.append((idx, total, status))

    results = run_batch_pipeline(batch, mode="lunar", progress_callback=callback)

    assert len(results) == 3
    # Item 1 succeeded
    assert results[0]["status"] == "done"
    assert results[0]["result"]["confidence_tag"] == "experimental_unverified"

    # Item 2 failed with isolated error
    assert results[1]["status"] == "error"
    assert "Query cannot be empty" in results[1]["error"]

    # Item 3 succeeded despite item 2 failure!
    assert results[2]["status"] == "done"
    assert results[2]["result"]["confidence_tag"] == "experimental_unverified"

    # Verify progress callback was invoked
    assert len(progress_calls) >= 3


def test_lunar_domain_guard_rejects_earth_queries():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    # Earth-specific query on lunar mode
    res = run_lunar_pipeline([img], [{"filename": "ch2_ohrc.tif"}], "Find buildings and roads in this scene")
    assert "Lunar mode is active" in res["answer"]
    assert "disabled" in res["answer"]
    assert "craters" in res["answer"]
    assert res["confidence_tag"] == "experimental_unverified"
    assert res["trace"]["parameters"]["domain_guard"] == "earth_keyword_intercepted"


def test_lunar_scientific_measurement_honesty():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    # 1. Uncalibrated image (no resolution in metadata)
    res_uncal = run_lunar_pipeline([img], [{"filename": "ch2_raw.tif"}], "Measure crater diameter")
    meas_uncal = res_uncal["verified_facts"]["measurements"]
    assert "unavailable" in meas_uncal["diameter"].lower()
    assert meas_uncal["scale_status"] == "uncalibrated"

    # 2. Calibrated image (resolution provided in metadata)
    res_cal = run_lunar_pipeline([img], [{"filename": "ch2_calibrated.tif", "resolution_m_per_pixel": 0.25}], "Measure crater diameter")
    meas_cal = res_cal["verified_facts"]["measurements"]
    assert "0.25" in meas_cal["diameter"]
    assert meas_cal["scale_status"] == "calibrated"


def test_lunar_report_agreement_is_not_disagreement():
    """Bug A: ensure skipped cross-verification renders as Not Cross-Verified, NOT Disagreement."""
    from satquery.pipeline.report_generator import generate_report

    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    res = run_lunar_pipeline([img], [{"filename": "test_lunar.png"}], "Analyze crater rims and shadows")

    report = generate_report(
        trace=res["trace"],
        phrased_answer=res["answer"],
        verified_facts=res["verified_facts"],
    )

    # Assert cross_verification parameter exists in trace
    assert res["trace"]["parameters"]["cross_verification"] == "skipped_no_deterministic_signal"

    # Assert Agreement is NOT Disagreement
    assert "- **Agreement:** Disagreement" not in report
    assert "- **Agreement:** Not Cross-Verified" in report or "- **Agreement:** Not Applicable" in report


def test_optical_sar_fusion_executive_summary_complete_sentence():
    """Bug B: ensure optical_sar_fusion produces complete sentences without raw concatenation or dangling parens."""
    from satquery.fusion.optical_sar_fusion import format_fusion_summary

    mock_res = {
        "land_cover_call": "water",
        "cloud_fraction": 0.746,
        "fractions": {"water": 0.510, "built_up": 0.200, "vegetation": 0.100},
    }
    summary = format_fusion_summary(mock_res)

    # Assert complete sentence structure
    assert summary.startswith("Optical-SAR fusion detected open water covering 51.0% of the scene.")
    assert "Optical imagery was 74.6% cloud-obscured" in summary
    assert "Sentinel-1 SAR backscatter" in summary
    # Assert no dangling double parenthesis
    assert not summary.endswith(".)")
    assert not summary.endswith("))")
    assert summary.endswith(".")
    assert "water (" not in summary

