"""Execution trace builder.

Produces the auditable execution summary required by the SIH 2026 problem
statement.  The returned dict is the **single most important contract in
this codebase** — any change to its shape is a breaking change requiring
team-wide notification.

Public API
----------
build_trace(task, tools_invoked, parameters, confidence, timestamp) -> dict
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_trace(
    task: str,
    tools_invoked: list[str],
    parameters: dict[str, Any],
    confidence: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build an execution trace conforming to the fixed contract.

    Parameters
    ----------
    task : str
        The ``TaskType.value`` string (e.g. ``"single_vqa"``).
    tools_invoked : list[str]
        Ordered list of specialist / tool names that were called.
    parameters : dict[str, Any]
        Key-value pairs summarising the parameters used.  All values
        must be JSON-serializable (no numpy arrays, no custom objects).
    confidence : str
        Overall confidence as a string (e.g. ``"0.95"``).
    timestamp : str | None
        ISO 8601 timestamp string.  If ``None``, the current UTC time
        is used.

    Returns
    -------
    dict
        Exactly::

            {
                "task": str,
                "tools_invoked": [str, ...],
                "parameters": {str: Any},
                "confidence": str,
                "timestamp": str   # ISO 8601
            }

    Raises
    ------
    TypeError
        If any value is not JSON-serializable.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    trace: dict[str, Any] = {
        "task": str(task),
        "tools_invoked": [str(t) for t in tools_invoked],
        "parameters": _ensure_serializable(parameters),
        "confidence": str(confidence),
        "timestamp": str(timestamp),
    }

    # Belt-and-suspenders: verify JSON round-trip
    import json
    try:
        json.dumps(trace)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Execution trace contains non-JSON-serializable values: {exc}"
        ) from exc

    return trace


def _ensure_serializable(obj: Any) -> Any:
    """Recursively coerce common non-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    # numpy scalars, enums, etc.
    return str(obj)
