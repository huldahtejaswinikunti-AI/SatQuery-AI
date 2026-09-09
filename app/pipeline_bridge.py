"""Pipeline bridge -- single adapter between the Streamlit UI and the backend.

Calls validate_input() -> route() -> execute() and normalises the output
into the contract shape that Person 5's UI expects::

    {
        "answer":   str,
        "overlay":  PIL.Image | None,
        "confidence": str,           # e.g. "0.94"
        "confidence_tag": str,       # e.g. "high_cross_verified"
        "trace":    dict,
        "report_path": str | None,
        # optional teammate additions -- always use .get() in the UI
        "consensus_score": float | None,
        "change_direction": str | None,
        "semantic_consistency": float | None,
        "validation_failure_reason": str | None,
    }

This is the ONLY file that imports from satquery.pipeline / satquery.router /
satquery.validator.  The rest of app/ imports only from here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Timeout for the full pipeline execution (seconds)
PIPELINE_TIMEOUT_SECONDS = 120


def run_pipeline(
    images: list,
    metas_or_query: list[dict] | str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Execute the full SatQuery AI pipeline and return a normalised result.

    Supports both signatures:
      run_pipeline(images: list, query: str)
      run_pipeline(images: list, metas: list[dict], query: str)
    """
    if isinstance(metas_or_query, str):
        q = metas_or_query
        m = [{} for _ in images]
    else:
        m = metas_or_query if metas_or_query is not None else [{} for _ in images]
        q = query or ""

    try:
        return _run_pipeline_inner(images, m, q)
    except Exception as exc:
        logger.exception("Pipeline failed")
        return _error_result(str(exc))


def _run_pipeline_inner(
    images: list,
    metas: list[dict],
    query: str,
) -> dict[str, Any]:
    """Inner implementation -- may raise."""
    from satquery.validator.input_validator import validate_input
    from satquery.validator.schemas import ValidationError as VError
    from satquery.router.task_router import route
    from satquery.pipeline.executor import execute, ExecutionError
    from satquery.pipeline.report_generator import generate_report

    # ---- Step 1: Validate -----------------------------------------------
    from satquery.pipeline.executor import cache_image_array
    image_dicts = []
    for idx, (img_arr, meta) in enumerate(zip(images, metas)):
        d = dict(meta)
        # Ensure required keys exist for the validator
        if "path" not in d:
            d["path"] = d.get("filename", f"upload_{idx}.png")
        cache_image_array(str(d["path"]), img_arr)
        if "band_count" not in d:
            d["band_count"] = 1 if img_arr.ndim == 2 else (
                img_arr.shape[0] if img_arr.ndim == 3 and img_arr.shape[0] <= 16 else
                img_arr.shape[2] if img_arr.ndim == 3 else 1
            )
        if "width" not in d:
            d["width"] = img_arr.shape[1]
        if "height" not in d:
            d["height"] = img_arr.shape[0]
        image_dicts.append(d)

    # For paired images, ensure acquisition timestamps exist for bi-temporal evaluation
    if len(image_dicts) == 2:
        from datetime import datetime, timedelta
        for idx, d in enumerate(image_dicts):
            if not d.get("timestamp"):
                acq = d.get("acquisition_date")
                if acq:
                    try:
                        d["timestamp"] = datetime.fromisoformat(str(acq).split("T")[0])
                    except Exception:
                        d["timestamp"] = datetime(2022, 1, 1) + timedelta(days=365 * idx)
                else:
                    d["timestamp"] = datetime(2022, 1, 1) + timedelta(days=365 * idx)

    validated = validate_input(image_dicts, query)

    if isinstance(validated, VError):
        return _error_result(
            validated.message,
            validation_failure_reason=validated.message,
        )

    # ---- Step 2: Route --------------------------------------------------
    modalities = [m.modality.value for m in validated.images]
    input_config = {
        "image_count": len(validated.images),
        "modalities": modalities,
    }
    route_result = route(query, input_config)
    task = route_result.task
    router_confidence = route_result.confidence

    # ---- Step 3: Execute ------------------------------------------------
    t0 = time.time()
    try:
        raw = execute(task, validated, router_confidence)
    except ExecutionError as exc:
        return _error_result(
            f"Analysis failed: {exc.specialist} could not complete the "
            f"'{exc.task.value}' task after retrying. "
            f"Details: {exc.original}",
            validation_failure_reason=str(exc.original),
        )
    elapsed = time.time() - t0
    logger.info("Pipeline executed in %.1f s", elapsed)

    # ---- Step 4: Normalise output ---------------------------------------
    trace = raw.get("trace", {})
    answer_raw = raw.get("answer", "")
    verified_facts = raw.get("verified_facts", {})

    # The answer may be a string (from phrasing) or a dict
    if isinstance(answer_raw, dict):
        answer_text = answer_raw.get("text", answer_raw.get("answer", str(answer_raw)))
    else:
        answer_text = str(answer_raw) if answer_raw else "Analysis complete."

    # Confidence — use specialist's raw_confidence as primary, not just router
    raw_conf = None
    if isinstance(verified_facts, dict):
        raw_conf = verified_facts.get("raw_confidence")
    if raw_conf is not None:
        conf_val = float(raw_conf)
    else:
        conf_str = trace.get("confidence", str(router_confidence))
        try:
            conf_val = float(conf_str)
        except (ValueError, TypeError):
            conf_val = router_confidence
    conf_str = str(round(conf_val, 4))

    # Determine confidence tag
    confidence_tag = _compute_confidence_tag(conf_val, verified_facts, task.value)

    # Overlay (may be provided by specialists)
    overlay = raw.get("overlay", verified_facts.get("overlay") if isinstance(verified_facts, dict) else None)

    # Generate markdown report with enriched verified facts
    report_md = generate_report(trace, answer_text, verified_facts=verified_facts)

    result = {
        "answer": answer_text,
        "overlay": overlay,
        "confidence": conf_str,
        "confidence_tag": confidence_tag,
        "confidence_score": conf_val,
        "trace": trace,
        "report_path": None,
        "report_markdown": report_md,
        "verified_facts": verified_facts,
        # Direct access to enriched specialist data
        "top_k": raw.get("top_k", verified_facts.get("top_k") if isinstance(verified_facts, dict) else None),
        "spectral_summary": raw.get("spectral_summary", verified_facts.get("spectral_summary") if isinstance(verified_facts, dict) else None),
        # Passthrough optional fields from teammates
        "consensus_score": raw.get("consensus_score",
                                    verified_facts.get("consensus_score") if isinstance(verified_facts, dict) else None),
        "change_direction": raw.get("change_direction",
                                     verified_facts.get("change_direction") if isinstance(verified_facts, dict) else None),
        "semantic_consistency": raw.get("semantic_consistency",
                                         verified_facts.get("semantic_consistency") if isinstance(verified_facts, dict) else None),
        "validation_failure_reason": None,
    }
    return result


def _compute_confidence_tag(
    confidence: float,
    verified_facts: Any,
    task_value: str,
) -> str:
    """Map confidence + verification status to a display tag."""
    if isinstance(verified_facts, dict):
        # Check for explicit tag from the cross-verifier
        explicit_tag = verified_facts.get("confidence_tag")
        if explicit_tag:
            return explicit_tag

        is_agreed = verified_facts.get("agreed", False)
        is_disagreement = verified_facts.get("disagreement", False)

        if is_disagreement:
            return "lower_confidence_disagreement"

        if is_agreed:
            if confidence >= 0.75:
                return "high_cross_verified"
            elif confidence >= 0.5:
                return "moderate"
            else:
                return "low"

    if confidence >= 0.85:
        return "high_rule_based"

    if confidence >= 0.5:
        return "moderate"

    return "lower_confidence"


def _error_result(
    message: str,
    validation_failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build a normalised error result dict."""
    return {
        "answer": message,
        "overlay": None,
        "confidence": "0.0",
        "confidence_tag": "error",
        "confidence_score": 0.0,
        "trace": {},
        "report_path": None,
        "report_markdown": "",
        "verified_facts": {},
        "consensus_score": None,
        "change_direction": None,
        "semantic_consistency": None,
        "validation_failure_reason": validation_failure_reason or message,
    }


# ---------------------------------------------------------------------------
# Lunar Pipeline Bridge (Task 3)
# ---------------------------------------------------------------------------


def run_lunar_pipeline(
    images: list,
    metas: list[dict] | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Execute the zero-shot Lunar analysis pipeline for Chandrayaan-2 imagery."""
    m = metas if metas is not None else [{} for _ in images]
    try:
        from satquery.lunar.lunar_pipeline import run_lunar_pipeline as _lunar_exec
        return _lunar_exec(images, m, query)
    except Exception as exc:
        logger.exception("Lunar pipeline execution failed")
        return {
            "answer": f"Lunar analysis error: {exc}",
            "overlay": None,
            "confidence": "experimental_unverified",
            "confidence_tag": "error",
            "confidence_score": None,
            "trace": {},
            "report_path": None,
            "report_markdown": "",
            "verified_facts": {},
            "validation_failure_reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Sequential Batch Queue Processor with Error Isolation (Task 2)
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    items: list[dict[str, Any]],
    mode: str = "earth",
    progress_callback: Any = None,
) -> list[dict[str, Any]]:
    """Process a queue of input groups sequentially with per-item error isolation.

    Parameters
    ----------
    items : list[dict]
        Each item has keys: 'images' (list), 'metas' (list), 'query' (str).
    mode : str
        'earth' or 'lunar'.
    progress_callback : callable, optional
        Callback invoked after each item: (idx, total, status, result_dict)

    Returns
    -------
    list[dict]
        Updated items list with 'status' ('done' or 'error') and 'result' or 'error'.
    """
    total = len(items)
    results = []

    for idx, it in enumerate(items):
        imgs = it.get("images", [])
        metas = it.get("metas", [])
        q = it.get("query", "")

        if progress_callback:
            progress_callback(idx, total, "running", None)

        try:
            if mode == "lunar":
                res = run_lunar_pipeline(imgs, metas, q)
            else:
                res = run_pipeline(imgs, metas, q)

            if res.get("confidence_tag") == "error":
                item_rec = {
                    **it,
                    "status": "error",
                    "error": res.get("validation_failure_reason") or res.get("answer", "Unknown error"),
                    "result": None,
                }
            else:
                item_rec = {
                    **it,
                    "status": "done",
                    "result": res,
                    "error": None,
                }
        except Exception as exc:
            logger.warning("Batch item #%d failed with exception: %s", idx + 1, exc)
            item_rec = {
                **it,
                "status": "error",
                "error": str(exc),
                "result": None,
            }

        results.append(item_rec)
        if progress_callback:
            progress_callback(idx, total, item_rec["status"], item_rec.get("result"))

    return results

