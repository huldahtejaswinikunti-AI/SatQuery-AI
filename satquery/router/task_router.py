"""Deterministic, rule-based task router.

Classifies a (query, input_config) pair into a ``TaskType`` using keyword /
phrase matching and input metadata — **zero model calls**.

Public API
----------
route(query, input_config) -> tuple[TaskType, float]
    Returns the matched task type and a confidence score (0.0–1.0).
"""

from __future__ import annotations

import re
from typing import Any

from satquery.router.task_types import TaskType


# ---------------------------------------------------------------------------
# Keyword / phrase banks — order matters (checked top-to-bottom)
# ---------------------------------------------------------------------------

# Strong keywords → confidence 1.0 when matched
_CHANGE_KEYWORDS_STRONG: list[str] = [
    "what changed",
    "what has changed",
    "before and after",
    "change detection",
    "temporal change",
    "changes between",
    "difference between",
    "how has .* changed",
]

# Weaker signals → confidence 0.7
_CHANGE_KEYWORDS_WEAK: list[str] = [
    "compared to",
    "over time",
    "evolution",
    "difference",
    "changed",
    "change",
]

_GROUNDING_KEYWORDS_STRONG: list[str] = [
    "where is",
    "where are",
    "highlight",
    "bounding box",
    "show me the location",
    "point to",
    "locate the",
    "localize",
]

_GROUNDING_KEYWORDS_WEAK: list[str] = [
    "locate",
    "mark",
    "show me",
    "find the",
    "identify the location",
]

_CAPTION_KEYWORDS_STRONG: list[str] = [
    "describe this image",
    "describe the image",
    "caption this",
    "generate a caption",
    "what is in this image",
    "summarize the scene",
    "tell me about this image",
]

_CAPTION_KEYWORDS_WEAK: list[str] = [
    "describe",
    "caption",
    "summarize",
    "overview of",
    "tell me about",
]


def _matches(text: str, patterns: list[str]) -> bool:
    """Return True if *text* matches any pattern (supports simple regex)."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RouteResult(tuple):
    """Result of route() that can be unpacked as (task, confidence)
    or used directly as a TaskType (equality with TaskType and str supported).
    """

    def __new__(cls, task: TaskType, confidence: float):
        return super().__new__(cls, (task, float(confidence)))

    @property
    def task(self) -> TaskType:
        return self[0]

    @property
    def confidence(self) -> float:
        return self[1]

    @property
    def value(self) -> str:
        return self[0].value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaskType):
            return self[0] == other
        if isinstance(other, str):
            return self[0].value == other or self[0] == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self[0])

    def __repr__(self) -> str:
        return f"RouteResult(task={self[0]!r}, confidence={self[1]})"


def route(
    query: str,
    input_config: dict[str, Any],
) -> RouteResult:
    """Route a query to a ``TaskType`` deterministically.

    Parameters
    ----------
    query : str
        User's natural-language question.
    input_config : dict
        Must contain:
        - ``image_count`` (int): number of images (1 or 2).
        - ``modalities`` (list[str]): per-image modality strings
          (``"optical"``, ``"sar"``, ``"unknown"``).

    Returns
    -------
    RouteResult
        Can be unpacked as ``(task, confidence)`` or used directly
        as a ``TaskType``.
        Confidence scores:
        - 1.0 — strong keyword match.
        - 0.7 — weaker/partial keyword match.
        - 0.3 — fallback based on input shape alone.
    """
    q = query.strip().lower()
    image_count: int = input_config.get("image_count", 1)
    modalities: list[str] = input_config.get("modalities", [])

    # ------------------------------------------------------------------
    # 1. CHANGE_VQA — requires 2 images + temporal/change language
    # ------------------------------------------------------------------
    if image_count == 2:
        if _matches(q, _CHANGE_KEYWORDS_STRONG):
            return RouteResult(TaskType.CHANGE_VQA, 1.0)
        if _matches(q, _CHANGE_KEYWORDS_WEAK):
            return RouteResult(TaskType.CHANGE_VQA, 0.7)

    # ------------------------------------------------------------------
    # 2. OPTICAL_SAR_FUSION — 2 images, one optical + one SAR
    # ------------------------------------------------------------------
    if image_count == 2:
        modality_set = set(modalities)
        if "optical" in modality_set and "sar" in modality_set:
            return RouteResult(TaskType.OPTICAL_SAR_FUSION, 1.0)

    # ------------------------------------------------------------------
    # 3. GROUNDING — spatial keywords (any image count)
    # ------------------------------------------------------------------
    if _matches(q, _GROUNDING_KEYWORDS_STRONG):
        return RouteResult(TaskType.GROUNDING, 1.0)
    if _matches(q, _GROUNDING_KEYWORDS_WEAK):
        return RouteResult(TaskType.GROUNDING, 0.7)

    # ------------------------------------------------------------------
    # 4. SINGLE_CAPTION — captioning keywords
    # ------------------------------------------------------------------
    if _matches(q, _CAPTION_KEYWORDS_STRONG):
        return RouteResult(TaskType.SINGLE_CAPTION, 1.0)
    if _matches(q, _CAPTION_KEYWORDS_WEAK):
        return RouteResult(TaskType.SINGLE_CAPTION, 0.7)

    # ------------------------------------------------------------------
    # 5. Fallback — SINGLE_VQA (default for single-image questions)
    # ------------------------------------------------------------------
    if image_count == 2:
        # Two images but no change/fusion signal — low confidence
        return RouteResult(TaskType.CHANGE_VQA, 0.3)

    return RouteResult(TaskType.SINGLE_VQA, 0.3)
