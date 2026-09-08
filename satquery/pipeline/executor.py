"""Pipeline executor — dispatches to specialist modules and orchestrates
cross-verification and phrasing.

Specialist imports are **stubbed** with clearly-named function calls and
comments marking which teammate owns each one.  Replace the stub imports
with real ones once teammates push their modules.

Exception policy: retry once, then fail loudly (``ExecutionError``).

Public API
----------
execute(task, validated_input) -> dict
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from satquery.pipeline.execution_trace import build_trace
from satquery.router.task_types import TaskType
from satquery.utils.config import SPECIALIST_MAX_RETRIES
from satquery.validator.schemas import ValidatedInput

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when a specialist fails after exhausting retries."""

    def __init__(self, task: TaskType, specialist: str, original: Exception):
        self.task = task
        self.specialist = specialist
        self.original = original
        super().__init__(
            f"Specialist '{specialist}' failed for task '{task.value}' "
            f"after {SPECIALIST_MAX_RETRIES + 1} attempt(s): {original}"
        )


# ---------------------------------------------------------------------------
# Specialist dispatch table
# ---------------------------------------------------------------------------
# Each entry maps TaskType -> (module_function, display_name, teammate)
# Replace stubs with real imports once teammates push their code.
# ---------------------------------------------------------------------------

def _stub_specialist(name: str):
    """Create a stub callable that raises NotImplementedError."""
    def _fn(validated_input: ValidatedInput) -> dict[str, Any]:
        raise NotImplementedError(
            f"Specialist '{name}' is not yet implemented. "
            "Replace this stub with the real import."
        )
    _fn.__qualname__ = name
    return _fn


# --- Teammate A: VQA specialist ---
try:
    from satquery.specialists.vqa import run_vqa  # type: ignore[import]
except ImportError:
    run_vqa = _stub_specialist("satquery.specialists.vqa.run_vqa")

# --- Teammate B: Captioning specialist ---
try:
    from satquery.specialists.captioning import run_caption  # type: ignore[import]
except ImportError:
    run_caption = _stub_specialist("satquery.specialists.captioning.run_caption")

# --- Teammate C: Grounding specialist ---
try:
    from satquery.specialists.grounding import run_grounding  # type: ignore[import]
except ImportError:
    run_grounding = _stub_specialist("satquery.specialists.grounding.run_grounding")

# --- Teammate D: Change detection specialist ---
try:
    from satquery.specialists.change_detection import run_change_vqa  # type: ignore[import]
except ImportError:
    run_change_vqa = _stub_specialist(
        "satquery.specialists.change_detection.run_change_vqa"
    )

# --- Teammate E: Fusion specialist ---
try:
    from satquery.specialists.fusion import run_fusion  # type: ignore[import]
except ImportError:
    run_fusion = _stub_specialist("satquery.specialists.fusion.run_fusion")

# --- Teammate F: Cross-verification ---
try:
    from satquery.cross_verification.verifier import verify  # type: ignore[import]
except ImportError:
    verify = _stub_specialist("satquery.cross_verification.verifier.verify")


def _get_specialist(task: TaskType) -> tuple[Any, str]:
    dispatch: dict[TaskType, tuple[Any, str]] = {
        TaskType.SINGLE_VQA: (globals().get("run_vqa", run_vqa), "vqa"),
        TaskType.SINGLE_CAPTION: (globals().get("run_caption", run_caption), "captioning"),
        TaskType.GROUNDING: (globals().get("run_grounding", run_grounding), "grounding"),
        TaskType.CHANGE_VQA: (globals().get("run_change_vqa", run_change_vqa), "change_detection"),
        TaskType.OPTICAL_SAR_FUSION: (globals().get("run_fusion", run_fusion), "fusion"),
    }
    return dispatch[task]


# ---------------------------------------------------------------------------
# Phrasing
# ---------------------------------------------------------------------------

try:
    from satquery.phrasing.phrasing_llm import phrase  # type: ignore[import]
except ImportError:
    phrase = _stub_specialist("satquery.phrasing.phrasing_llm.phrase")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute(
    task: TaskType,
    validated_input: ValidatedInput,
    router_confidence: float = 1.0,
) -> dict[str, Any]:
    """Run the full pipeline for a given task and validated input.

    Parameters
    ----------
    task : TaskType
        The routed task type.
    validated_input : ValidatedInput
        The validated input from the validator layer.
    router_confidence : float
        Confidence returned by the router (included in the trace).

    Returns
    -------
    dict
        Keys: ``trace``, ``answer``, ``verified_facts``.

    Raises
    ------
    ExecutionError
        If the specialist fails after retrying.
    """
    tools_invoked: list[str] = []
    parameters: dict[str, Any] = {
        "image_count": len(validated_input.images),
        "input_type": validated_input.input_type.value,
        "query": validated_input.query,
    }
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Step 1: Call specialist ---
    specialist_fn, specialist_name = _get_specialist(task)
    tools_invoked.append(specialist_name)

    facts = _call_with_retry(
        specialist_fn, validated_input, task, specialist_name,
    )

    # --- Step 2: Cross-verification ---
    tools_invoked.append("cross_verification")
    verified_facts = _call_with_retry(
        lambda inp: globals().get("verify", verify)(facts),  # verify takes facts, not input
        validated_input,
        task,
        "cross_verification",
    )

    # --- Step 3: Phrasing ---
    tools_invoked.append("phrasing_llm")
    answer = _call_with_retry(
        lambda inp: globals().get("phrase", phrase)(verified_facts),
        validated_input,
        task,
        "phrasing_llm",
    )

    # --- Step 4: Build trace ---
    trace = build_trace(
        task=task.value,
        tools_invoked=tools_invoked,
        parameters=parameters,
        confidence=str(router_confidence),
        timestamp=timestamp,
    )

    return {
        "trace": trace,
        "answer": answer,
        "verified_facts": verified_facts,
    }


def _call_with_retry(
    fn,
    validated_input: ValidatedInput,
    task: TaskType,
    name: str,
) -> Any:
    """Call *fn* with retry logic per the configured policy."""
    last_exc: Exception | None = None

    for attempt in range(SPECIALIST_MAX_RETRIES + 1):
        try:
            return fn(validated_input)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Specialist '%s' attempt %d/%d failed: %s",
                name,
                attempt + 1,
                SPECIALIST_MAX_RETRIES + 1,
                exc,
            )
            if attempt < SPECIALIST_MAX_RETRIES:
                time.sleep(0.5)  # brief back-off before retry

    raise ExecutionError(task, name, last_exc)  # type: ignore[arg-type]
