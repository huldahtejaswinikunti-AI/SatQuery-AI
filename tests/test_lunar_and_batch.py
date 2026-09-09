"""Tests for Chandrayaan-2 Lunar Mode and Multi-Input Batch Processing."""
from __future__ import annotations

import numpy as np
import pytest

from satquery.lunar.lunar_validator import validate_lunar_input
from satquery.lunar.lunar_pipeline import run_lunar_pipeline
from app.pipeline_bridge import run_lunar_pipeline as bridge_run_lunar, run_batch_pipeline
from app.ui_components import _BADGE_CONFIG


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
