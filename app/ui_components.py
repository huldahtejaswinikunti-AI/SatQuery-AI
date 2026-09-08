"""Reusable Streamlit UI widgets for SatQuery AI.

Every widget renders based on what data is present -- optional fields
(consensus_score, change_direction, semantic_consistency) are shown only
when non-None, and hidden silently otherwise.

Accessibility: confidence badges use colour AND icon/label, never colour
alone.  All interactive elements have visible focus states via CSS.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit as st

from satquery.utils.image_utils import to_display_rgb


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Render the app header with logo and subtitle."""
    logo_col, title_col = st.columns([0.08, 0.92])
    from pathlib import Path

    logo_path = Path(__file__).parent / "assets" / "logo.png"
    with logo_col:
        if logo_path.exists():
            st.image(str(logo_path), width=48)
    with title_col:
        st.markdown(
            '<h1 class="main-header">SatQuery AI</h1>'
            '<p class="main-subtitle">'
            "Agentic Vision-Language Assistant for Remote Sensing "
            "with Classical Signal Grounding<br>"
            '<span class="main-meta">SIH 2026 &middot; PS 26167 &middot; ISRO / SAC</span>'
            "</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Image preview
# ---------------------------------------------------------------------------


def render_image_preview(images: list[np.ndarray], metas: list[dict]) -> None:
    """Show uploaded / demo images with metadata captions."""
    if not images:
        st.info(
            "Upload 1 or 2 satellite images, or select a pre-loaded demo "
            "scenario from the sidebar."
        )
        return

    st.subheader("Input Imagery")

    if len(images) == 1:
        rgb = to_display_rgb(images[0])
        name = metas[0].get("filename", "Image 1") if metas else "Image 1"
        shape_str = f"{images[0].shape[1]}x{images[0].shape[0]}"
        st.image(rgb, caption=f"{name} ({shape_str})", use_container_width=True)
    else:
        cols = st.columns(2)
        for i, (img, meta) in enumerate(zip(images, metas)):
            with cols[i]:
                rgb = to_display_rgb(img)
                name = meta.get("filename", f"Image {i + 1}")
                shape_str = f"{img.shape[1]}x{img.shape[0]}"
                st.image(
                    rgb,
                    caption=f"{name} ({shape_str})",
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Confidence badge
# ---------------------------------------------------------------------------

_BADGE_CONFIG = {
    "high_cross_verified": {
        "cls": "badge-high-verified",
        "icon": "[VERIFIED]",
        "label": "High Confidence (Cross-Verified)",
    },
    "high_sar_penetration": {
        "cls": "badge-high-verified",
        "icon": "[SAR-OK]",
        "label": "High Confidence (SAR-Penetrated)",
    },
    "high_rule_based": {
        "cls": "badge-high-rule",
        "icon": "[SIGNAL]",
        "label": "Deterministic Signal",
    },
    "lower_confidence_disagreement": {
        "cls": "badge-disagreement",
        "icon": "[!]",
        "label": "Lower Confidence (Signal Disagreement)",
    },
    "moderate": {
        "cls": "badge-unverified",
        "icon": "[~]",
        "label": "Moderate Confidence (Unverified)",
    },
    "error": {
        "cls": "badge-error",
        "icon": "[X]",
        "label": "Input Validation Error",
    },
}


def render_confidence_badge(result: dict[str, Any]) -> None:
    """Render the primary confidence badge with icon + label + percentage."""
    tag = result.get("confidence_tag", "moderate")
    score = result.get("confidence_score", 0.0)
    pct = int(round(float(score) * 100))

    cfg = _BADGE_CONFIG.get(tag, _BADGE_CONFIG["moderate"])

    if tag == "error":
        label = f'{cfg["icon"]} {cfg["label"]}'
    else:
        label = f'{cfg["icon"]} {cfg["label"]}: {pct}%'

    st.markdown(
        f'<div class="badge-container {cfg["cls"]}" role="status" '
        f'aria-label="{cfg["label"]}">{label}</div>',
        unsafe_allow_html=True,
    )

    # Second confidence signal: semantic consistency (only if present)
    sem = result.get("semantic_consistency")
    if sem is not None:
        sem_pct = int(round(float(sem) * 100))
        st.markdown(
            f'<div class="badge-container badge-semantic" role="status" '
            f'aria-label="Semantic consistency">'
            f"[SEMANTIC] Semantic Consistency: {sem_pct}%</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Consensus score (headline metric when present)
# ---------------------------------------------------------------------------


def render_consensus_score(result: dict[str, Any]) -> None:
    """Show optical-SAR consensus as a headline metric, or nothing at all."""
    score = result.get("consensus_score")
    if score is None:
        return
    pct = int(round(float(score)))
    st.markdown(
        f'<div class="consensus-metric" role="status">'
        f"Optical-SAR Agreement: <strong>{pct}%</strong></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Change direction
# ---------------------------------------------------------------------------


def render_change_direction(result: dict[str, Any]) -> None:
    """Show change direction sentence if present, skip otherwise."""
    direction = result.get("change_direction")
    if not direction:
        return
    st.markdown(
        f'<div class="change-direction">'
        f"Change concentrated in the <strong>{direction}</strong></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overlay display
# ---------------------------------------------------------------------------


def render_overlay(result: dict[str, Any]) -> None:
    """Show the visual grounding overlay if present."""
    overlay = result.get("overlay")
    if overlay is None:
        return
    st.image(
        overlay,
        caption="Visual Grounding & Analysis Overlay",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


def render_execution_trace(result: dict[str, Any]) -> None:
    """Expandable execution trace panel with JSON download."""
    trace = result.get("trace", {})
    if not trace:
        return

    with st.expander("Auditable Execution Trace & Tool Timeline", expanded=False):
        st.markdown(
            f"**Task:** `{trace.get('task', 'N/A')}` | "
            f"**Confidence:** `{trace.get('confidence', 'N/A')}` | "
            f"**Timestamp:** `{trace.get('timestamp', 'N/A')}`"
        )

        tools = trace.get("tools_invoked", [])
        if tools:
            st.markdown("**Tools invoked:** " + " -> ".join(f"`{t}`" for t in tools))

        params = trace.get("parameters", {})
        if params:
            st.markdown("**Parameters:**")
            for k, v in params.items():
                st.markdown(f"- `{k}`: {v}")

        # Show full JSON
        st.json(trace)


# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------


def render_download_buttons(result: dict[str, Any]) -> None:
    """Render download buttons for JSON trace, markdown report, and PDF."""
    cols = st.columns(3)

    # JSON trace
    trace_json = json.dumps(result.get("trace", {}), indent=2, default=str)
    with cols[0]:
        st.download_button(
            "Download JSON Trace",
            trace_json,
            "satquery_trace.json",
            "application/json",
            use_container_width=True,
        )

    # Markdown report
    report_md = result.get("report_markdown", "")
    if report_md:
        with cols[1]:
            st.download_button(
                "Download Markdown Report",
                report_md,
                "satquery_report.md",
                "text/markdown",
                use_container_width=True,
            )

    # PDF report (optional -- only if fpdf2 is available)
    with cols[2]:
        try:
            try:
                from app.pdf_report import generate_pdf_report
            except ImportError:
                from pdf_report import generate_pdf_report

            pdf_bytes = generate_pdf_report(
                result,
                result.get("trace", {}).get("parameters", {}).get("query", ""),
                [],
            )
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                "satquery_report.pdf",
                "application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                "Download PDF Report",
                trace_json,
                "satquery_report.json",
                "application/json",
                use_container_width=True,
                disabled=True,
                help="PDF generation not available (fpdf2 not installed)",
            )
