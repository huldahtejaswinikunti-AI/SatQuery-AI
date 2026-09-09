"""Tests for satquery.pipeline.executor — mock-based, no real models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from satquery.pipeline.executor import ExecutionError, execute
from satquery.router.task_types import TaskType
from satquery.validator.schemas import (
    ImageMeta,
    InputType,
    Modality,
    ValidatedInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(
    image_count: int = 1,
    input_type: InputType = InputType.SINGLE,
    query: str = "How many buildings?",
) -> ValidatedInput:
    """Build a minimal ValidatedInput for testing."""
    images = []
    for i in range(image_count):
        images.append(
            ImageMeta(
                path=f"/fake/image_{i}.tif",
                format="GeoTIFF",
                band_count=3,
                modality=Modality.OPTICAL,
                crs=4326,
                width=64,
                height=64,
            )
        )
    return ValidatedInput(
        images=images,
        input_type=input_type,
        query=query,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Verify the correct specialist is called for each TaskType."""

    @pytest.mark.parametrize(
        "task, specialist_path",
        [
            (TaskType.SINGLE_VQA, "satquery.pipeline.executor.run_vqa"),
            (TaskType.SINGLE_CAPTION, "satquery.pipeline.executor.run_caption"),
            (TaskType.GROUNDING, "satquery.pipeline.executor.run_grounding"),
            (TaskType.CHANGE_VQA, "satquery.pipeline.executor.run_change_vqa"),
            (TaskType.OPTICAL_SAR_FUSION, "satquery.pipeline.executor.run_fusion"),
        ],
    )
    @patch("satquery.pipeline.executor.phrase", return_value="Phrased answer.")
    @patch("satquery.pipeline.executor.verify", return_value={"verified": True})
    def test_calls_correct_specialist(
        self, mock_verify, mock_phrase, task, specialist_path,
    ):
        mock_specialist = MagicMock(return_value={"result": "test"})

        with patch(specialist_path, mock_specialist):
            result = execute(task, _make_input())

        mock_specialist.assert_called_once()
        mock_verify.assert_called_once_with({"result": "test"})
        # phrase is called with enriched verified_facts (includes propagated fields)
        mock_phrase.assert_called_once()
        phrase_arg = mock_phrase.call_args[0][0]
        assert phrase_arg["verified"] is True

        assert "trace" in result
        assert "answer" in result
        assert result["answer"] == "Phrased answer."


class TestRetryPolicy:
    """Specialist failure → retry once → fail loudly."""

    @patch("satquery.pipeline.executor.phrase", return_value="answer")
    @patch("satquery.pipeline.executor.verify", return_value={})
    def test_retry_then_fail(self, mock_verify, mock_phrase):
        failing_fn = MagicMock(side_effect=RuntimeError("boom"))

        with patch("satquery.pipeline.executor.run_vqa", failing_fn):
            with pytest.raises(ExecutionError) as exc_info:
                execute(TaskType.SINGLE_VQA, _make_input())

        # Should have been called twice: initial + 1 retry
        assert failing_fn.call_count == 2
        assert "boom" in str(exc_info.value)

    @patch("satquery.pipeline.executor.phrase", return_value="answer")
    @patch("satquery.pipeline.executor.verify", return_value={})
    def test_succeeds_on_retry(self, mock_verify, mock_phrase):
        call_count = {"n": 0}

        def flaky_fn(inp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return {"recovered": True}

        with patch("satquery.pipeline.executor.run_vqa", flaky_fn):
            result = execute(TaskType.SINGLE_VQA, _make_input())

        assert result["answer"] == "answer"
        assert call_count["n"] == 2


class TestTraceInOutput:
    """Verify the trace dict is included in the output."""

    @patch("satquery.pipeline.executor.phrase", return_value="answer")
    @patch("satquery.pipeline.executor.verify", return_value={})
    @patch("satquery.pipeline.executor.run_vqa", return_value={"x": 1})
    def test_trace_present(self, *_):
        result = execute(TaskType.SINGLE_VQA, _make_input(), router_confidence=0.9)
        trace = result["trace"]

        assert trace["task"] == "single_vqa"
        assert isinstance(trace["tools_invoked"], list)
        assert trace["confidence"] == "0.9"
        assert "timestamp" in trace
