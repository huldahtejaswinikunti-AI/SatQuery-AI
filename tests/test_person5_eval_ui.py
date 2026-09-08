"""Automated tests for Person 5 deliverables.

Verifies:
- evaluation.eval_utils (BLEU-4, METEOR, accuracy, IoU, confusion matrix, F1)
- app.pdf_report (generate_pdf_report)
- app.pipeline_bridge (bridge output normalization & timeout resilience)
- tests/fixtures (sample_trace.json, sample_pipeline_result.json schema validation)
- data/demo_samples/metadata.json catalog validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.pdf_report import generate_pdf_report
from app.pipeline_bridge import run_pipeline
from evaluation.eval_utils import (
    compute_accuracy,
    compute_bleu_1,
    compute_bleu_4,
    compute_confusion_matrix,
    compute_f1_score,
    compute_iou,
    compute_mask_iou,
    compute_meteor,
    compute_token_f1,
    normalize_text,
)


def test_text_normalization():
    assert normalize_text("  Water Body, 100%!  ") == "water body 100"
    assert normalize_text("") == ""


def test_token_f1():
    assert compute_token_f1("deep water body", "deep water body") == 1.0
    assert compute_token_f1("deep water body", "shallow water body") > 0.6
    assert compute_token_f1("urban building", "water lake") == 0.0


def test_bleu_metrics():
    ref = "coastal port with shipping docks and open sea water"
    hyp = "coastal port with container shipping docks and water"
    b1 = compute_bleu_1(ref, hyp)
    b4 = compute_bleu_4(ref, hyp)
    assert 0.0 < b1 <= 1.0
    assert 0.0 < b4 <= 1.0
    assert compute_bleu_4("", "") == 0.0


def test_meteor_metric():
    ref = "severe flood inundation across agricultural plains"
    hyp = "flood inundation across plains"
    score = compute_meteor(ref, hyp)
    assert 0.5 < score <= 1.0
    assert compute_meteor("", "") == 0.0


def test_accuracy_metric():
    preds = ["urban area", "water body", "agricultural land"]
    gts = ["urban fabric", "water body", "cropland"]
    acc = compute_accuracy(preds, gts)
    assert acc > 0.3  # at least the exact match "water body" matches


def test_mask_iou_and_confusion():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    iou = compute_mask_iou(pred, gt)
    assert pytest.approx(iou, 0.01) == 1.0 / 3.0

    cm = compute_confusion_matrix(pred, gt)
    assert cm["tp"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 1

    f1 = compute_f1_score(precision=0.5, recall=0.5)
    assert pytest.approx(f1, 0.01) == 0.5


def test_generate_pdf_report():
    dummy_result = {
        "answer": "Industrial logistics center with container berths.",
        "confidence_tag": "VERIFIED_HIGH",
        "raw_confidence": 0.94,
        "consensus_score": 0.94,
        "semantic_consistency": "AGREE",
        "trace": {
            "task": "single_vqa",
            "tools_invoked": ["InputValidator", "GeoChatSpecialist"],
            "execution_time_seconds": 1.45,
            "verified_facts": [{"fact": "Surface water verified", "verified": True}],
        },
    }
    images_meta = [{"bands": ["Red", "Green", "Blue"], "shape": [256, 256, 3], "sensor": "Sentinel-2"}]

    pdf_bytes = generate_pdf_report(dummy_result, "Describe this facility", images_meta)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_pipeline_bridge_graceful_handling():
    # Calling run_pipeline with empty images should gracefully return validation failure without crashing
    res = run_pipeline([], "What is in this scene?")
    assert isinstance(res, dict)
    assert "answer" in res
    assert res["confidence_tag"] in ("error", "VALIDATION_FAILED")
    assert res.get("validation_failure_reason") is not None


def test_fixtures_schema():
    trace_path = Path("tests/fixtures/sample_trace.json")
    result_path = Path("tests/fixtures/sample_pipeline_result.json")

    assert trace_path.exists()
    assert result_path.exists()

    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
        assert "trace_id" in trace
        assert "task" in trace
        assert "tools_invoked" in trace

    with open(result_path, encoding="utf-8") as f:
        res = json.load(f)
        assert "answer" in res
        assert "confidence_tag" in res
        assert "trace" in res


def test_demo_catalog_metadata():
    meta_path = Path("data/demo_samples/metadata.json")
    assert meta_path.exists()

    with open(meta_path, encoding="utf-8") as f:
        catalog = json.load(f)
        assert "samples" in catalog
        assert len(catalog["samples"]) >= 16
