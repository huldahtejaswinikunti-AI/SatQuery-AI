"""Task-type enumeration for the SatQuery AI router.

Every routed query maps to exactly one ``TaskType``.  The set is fixed —
new values require a team-wide schema review.
"""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """Supported analysis tasks.

    Values are lowercase identifiers used in execution traces and JSON APIs.
    """

    SINGLE_VQA = "single_vqa"
    SINGLE_CAPTION = "single_caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
