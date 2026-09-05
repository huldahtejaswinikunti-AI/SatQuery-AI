import numpy as np
from satquery.pipeline.executor import PipelineExecutor

def test_executor_vqa():
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    res = PipelineExecutor().run([img], [{"filename": "test.png"}], "What is visible?")
    assert "answer" in res
    assert res["confidence_score"] > 0.0
    assert "trace" in res

def test_executor_grounding():
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    res = PipelineExecutor().run([img], [{"filename": "test.png"}], "Highlight the water")
    assert res["trace"]["task"] == "grounding"
    assert res["overlay"] is not None
