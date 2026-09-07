"""Tests for satquery.router.task_router — deterministic, zero model calls.

These tests MUST pass without importing transformers, torch, or any
model-loading code.
"""

from __future__ import annotations

import sys

import pytest

from satquery.router.task_router import route
from satquery.router.task_types import TaskType


class TestChangeVQA:
    """Two-image + temporal keywords → CHANGE_VQA."""

    @pytest.mark.parametrize(
        "query",
        [
            "What changed between these two images?",
            "Show me the before and after",
            "What is the difference between 2020 and 2024?",
            "How has the coastline changed over time?",
            "Detect changes between the two dates",
        ],
    )
    def test_strong_change_keywords(self, query):
        cfg = {"image_count": 2, "modalities": ["optical", "optical"]}
        task, confidence = route(query, cfg)
        assert task == TaskType.CHANGE_VQA
        assert confidence >= 0.7

    def test_weak_change_keyword(self):
        cfg = {"image_count": 2, "modalities": ["optical", "optical"]}
        task, confidence = route("compared to last year", cfg)
        assert task == TaskType.CHANGE_VQA
        assert confidence >= 0.7


class TestOpticalSarFusion:
    """Two images with one optical + one SAR → OPTICAL_SAR_FUSION."""

    def test_fusion_detection(self):
        cfg = {"image_count": 2, "modalities": ["optical", "sar"]}
        task, confidence = route("Analyse these images together", cfg)
        assert task == TaskType.OPTICAL_SAR_FUSION
        assert confidence == 1.0

    def test_fusion_reversed_modalities(self):
        cfg = {"image_count": 2, "modalities": ["sar", "optical"]}
        task, confidence = route("Merge sensor data", cfg)
        assert task == TaskType.OPTICAL_SAR_FUSION
        assert confidence == 1.0


class TestGrounding:
    """Spatial keywords → GROUNDING."""

    @pytest.mark.parametrize(
        "query",
        [
            "Where is the airport?",
            "Highlight all water bodies",
            "Locate the stadium",
            "Show me the location of the bridge",
            "Draw a bounding box around the reservoir",
        ],
    )
    def test_grounding_keywords(self, query):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        task, confidence = route(query, cfg)
        assert task == TaskType.GROUNDING
        assert confidence >= 0.7


class TestSingleCaption:
    """Captioning keywords → SINGLE_CAPTION."""

    @pytest.mark.parametrize(
        "query",
        [
            "Describe this image",
            "Generate a caption for this satellite image",
            "What is in this image?",
            "Summarize the scene",
            "Tell me about this image",
        ],
    )
    def test_caption_keywords(self, query):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        task, confidence = route(query, cfg)
        assert task == TaskType.SINGLE_CAPTION
        assert confidence >= 0.7


class TestSingleVQA:
    """Default fallback for single-image queries → SINGLE_VQA."""

    def test_generic_question(self):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        task, _ = route("How many buildings are there?", cfg)
        assert task == TaskType.SINGLE_VQA

    def test_counting_question(self):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        task, _ = route("Count the ships in the harbour", cfg)
        assert task == TaskType.SINGLE_VQA


class TestDeterminism:
    """Router must be deterministic — same input → same output."""

    def test_repeated_calls(self):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        results = [route("Where is the river?", cfg) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestRouteResultDualUsage:
    """Verify RouteResult works as tuple or directly as TaskType."""

    def test_direct_equality(self):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        result = route("Where is the airport?", cfg)
        assert result == TaskType.GROUNDING
        assert result == "grounding"
        assert result.task == TaskType.GROUNDING
        assert result.confidence == 1.0
        assert result.value == "grounding"

    def test_tuple_unpacking(self):
        cfg = {"image_count": 1, "modalities": ["optical"]}
        task, confidence = route("Where is the airport?", cfg)
        assert task == TaskType.GROUNDING
        assert confidence == 1.0


class TestNoModelImports:
    """Verify that running the router never imported model libraries."""

    def test_no_transformers_import(self):
        # If transformers was imported, it would be in sys.modules
        assert "transformers" not in sys.modules, (
            "Router must not import transformers — it must be model-free."
        )

    def test_no_torch_import(self):
        # torch may be imported by config.get_device(), but only if it was
        # already installed. The router itself must not trigger it.
        # We check that task_router module doesn't have torch in its imports.
        import satquery.router.task_router as mod
        source = open(mod.__file__).read()
        assert "import torch" not in source
        assert "from torch" not in source
