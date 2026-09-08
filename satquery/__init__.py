"""
SatQuery AI: Vision-Language Assistant with Deterministic Signal Grounding.
Built for SIH 2026 Problem Statement 26167 (ISRO/SAC).
"""
from satquery.pipeline.executor import execute, ExecutionError
from satquery.pipeline.execution_trace import build_trace
from satquery.pipeline.report_generator import generate_report
from satquery.validator.input_validator import validate_input
from satquery.router.task_router import route

__version__ = "0.1.0"
__all__ = [
    "execute", "ExecutionError", "build_trace",
    "generate_report", "validate_input", "route",
]
