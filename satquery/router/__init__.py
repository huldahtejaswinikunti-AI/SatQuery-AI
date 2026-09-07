"""Task routing for SatQuery AI."""

from satquery.router.task_router import RouteResult, route
from satquery.router.task_types import TaskType

__all__ = [
    "route",
    "RouteResult",
    "TaskType",
]


def __getattr__(name: str):
    """Lazy import for parse_intent to avoid eager model loading."""
    if name == "parse_intent":
        from satquery.router.intent_parser import parse_intent
        return parse_intent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
