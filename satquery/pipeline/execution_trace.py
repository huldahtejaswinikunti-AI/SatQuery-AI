from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

class ToolStep(BaseModel):
    tool_name: str
    status: str = "completed"
    duration_ms: float = 0.0
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)

class ExecutionTrace(BaseModel):
    trace_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task: str
    query: str
    tools_invoked: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    confidence_tag: str
    steps: list[ToolStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
