from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    SINGLE_VQA = "single_vqa"
    SINGLE_CAPTION = "single_caption"
    GROUNDING = "grounding"
    BITEMPORAL_CHANGE = "change_vqa"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"

class RoutingDecision(BaseModel):
    task: TaskType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    target_phrase: str | None = None
    invoked_tools: list[str] = Field(default_factory=list)
