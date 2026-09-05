import json
from satquery.pipeline.execution_trace import ExecutionTrace, ToolStep

def test_trace():
    t = ExecutionTrace(
        trace_id="t1", task="single_vqa", query="query",
        tools_invoked=["tool1"], parameters={}, confidence=0.9, confidence_tag="high",
        steps=[ToolStep(tool_name="tool1", duration_ms=2.5)]
    )
    data = json.loads(t.to_json())
    assert data["trace_id"] == "t1"
    assert data["steps"][0]["tool_name"] == "tool1"
