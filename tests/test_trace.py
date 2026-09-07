"""Tests for satquery.pipeline.execution_trace — schema contract tests.

The trace schema is the single most important contract in this codebase.
These tests enforce the exact shape documented in the spec.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from satquery.pipeline.execution_trace import build_trace


# ---------------------------------------------------------------------------
# Exact schema shape
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"task", "tools_invoked", "parameters", "confidence", "timestamp"}


class TestTraceSchema:
    """The trace dict must have exactly the documented keys and types."""

    def test_exact_keys(self):
        trace = build_trace(
            task="single_vqa",
            tools_invoked=["vqa"],
            parameters={"query": "test"},
            confidence="0.95",
            timestamp="2026-09-07T19:18:19+05:30",
        )
        assert set(trace.keys()) == _REQUIRED_KEYS

    def test_no_extra_keys(self):
        trace = build_trace(
            task="grounding",
            tools_invoked=["grounding"],
            parameters={},
            confidence="1.0",
        )
        assert set(trace.keys()) == _REQUIRED_KEYS

    def test_task_is_string(self):
        trace = build_trace("single_vqa", [], {}, "0.5")
        assert isinstance(trace["task"], str)

    def test_tools_invoked_is_list_of_strings(self):
        trace = build_trace("single_vqa", ["vqa", "verifier"], {}, "0.8")
        assert isinstance(trace["tools_invoked"], list)
        assert all(isinstance(t, str) for t in trace["tools_invoked"])

    def test_parameters_is_dict(self):
        trace = build_trace("single_vqa", [], {"key": "value"}, "0.9")
        assert isinstance(trace["parameters"], dict)

    def test_confidence_is_string(self):
        trace = build_trace("single_vqa", [], {}, "0.75")
        assert isinstance(trace["confidence"], str)
        assert trace["confidence"] == "0.75"

    def test_timestamp_is_string(self):
        trace = build_trace("single_vqa", [], {}, "1.0")
        assert isinstance(trace["timestamp"], str)


class TestJSONSerializable:
    """Trace must be fully JSON-serializable."""

    def test_json_dumps_succeeds(self):
        trace = build_trace(
            task="change_vqa",
            tools_invoked=["change_detection", "verifier"],
            parameters={"image_count": 2, "query": "what changed?"},
            confidence="0.85",
            timestamp="2026-09-07T19:18:19+05:30",
        )
        serialized = json.dumps(trace)
        assert isinstance(serialized, str)

    def test_json_roundtrip(self):
        trace = build_trace(
            task="optical_sar_fusion",
            tools_invoked=["fusion"],
            parameters={"modalities": ["optical", "sar"]},
            confidence="1.0",
        )
        roundtripped = json.loads(json.dumps(trace))
        assert roundtripped == trace


class TestTimestamp:
    """Timestamp handling."""

    def test_explicit_timestamp_preserved(self):
        ts = "2026-09-07T19:18:19+05:30"
        trace = build_trace("single_vqa", [], {}, "0.5", timestamp=ts)
        assert trace["timestamp"] == ts

    def test_auto_timestamp_is_valid_iso(self):
        trace = build_trace("single_vqa", [], {}, "0.5")
        # Should parse as ISO 8601 without error
        dt = datetime.fromisoformat(trace["timestamp"])
        assert isinstance(dt, datetime)


class TestEdgeCases:
    """Edge cases and coercion."""

    def test_empty_tools_list(self):
        trace = build_trace("single_vqa", [], {}, "0.0")
        assert trace["tools_invoked"] == []

    def test_empty_parameters(self):
        trace = build_trace("single_vqa", [], {}, "0.0")
        assert trace["parameters"] == {}

    def test_numeric_confidence_coerced_to_string(self):
        # If someone accidentally passes a float, it should be coerced
        trace = build_trace("single_vqa", [], {}, 0.95)  # type: ignore[arg-type]
        assert isinstance(trace["confidence"], str)
        assert trace["confidence"] == "0.95"

    def test_nested_parameters_serializable(self):
        params = {
            "images": ["/path/a.tif", "/path/b.tif"],
            "thresholds": {"ndvi": 0.3, "ndwi": 0.2},
        }
        trace = build_trace("change_vqa", ["cd"], params, "0.8")
        json.dumps(trace)  # must not raise
