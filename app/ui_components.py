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
        "icon": "✓",
        "label": "High Confidence (Cross-Verified)",
    },
    "high_sar_penetration": {
        "cls": "badge-high-verified",
        "icon": "✓",
        "label": "High Confidence (SAR-Penetrated)",
    },
    "high_rule_based": {
        "cls": "badge-high-rule",
        "icon": "⚡",
        "label": "Deterministic Signal",
    },
    "lower_confidence_disagreement": {
        "cls": "badge-disagreement",
        "icon": "!",
        "label": "Lower Confidence (Signal Disagreement)",
    },
    "lower_confidence": {
        "cls": "badge-disagreement",
        "icon": "!",
        "label": "Lower Confidence",
    },
    "moderate": {
        "cls": "badge-unverified",
        "icon": "~",
        "label": "Moderate Confidence (Unverified)",
    },
    "experimental_unverified": {
        "cls": "badge-experimental-unverified",
        "icon": "⊘",
        "label": "Experimental — No Cross-Check Available",
    },
    "error": {
        "cls": "badge-error",
        "icon": "✕",
        "label": "Input Validation Error",
    },
}


def render_confidence_badge(result: dict[str, Any]) -> None:
    """Render the primary confidence badge with icon + label + percentage."""
    tag = result.get("confidence_tag", "moderate")
    score = result.get("confidence_score")
    
    cfg = _BADGE_CONFIG.get(tag, _BADGE_CONFIG["moderate"])

    if tag == "experimental_unverified":
        label = f'{cfg["icon"]} {cfg["label"]}'
        st.markdown(
            f'<div class="badge-container {cfg["cls"]}" role="status" '
            f'aria-label="{cfg["label"]}">{label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="lunar-note-banner">'
            '<strong>Notice:</strong> Lunar imagery has no deterministic cross-check available in this system — '
            'treat this answer as unverified.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if tag == "error":
        label = f'{cfg["icon"]} {cfg["label"]}'
    elif score is not None:
        pct = int(round(float(score) * 100))
        label = f'{cfg["icon"]} {cfg["label"]}: {pct}%'
    else:
        label = f'{cfg["icon"]} {cfg["label"]}'

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


# ---------------------------------------------------------------------------
# Spectral Analysis Summary
# ---------------------------------------------------------------------------


def render_spectral_summary(result: dict[str, Any]) -> None:
    """Render spectral index bars and scene composition breakdown."""
    vf = result.get("verified_facts", {})
    if not isinstance(vf, dict):
        return
    spectral = vf.get("spectral_summary", {})
    if not spectral:
        return

    with st.expander("Spectral Analysis & Scene Composition", expanded=True):
        ndvi = spectral.get("ndvi_mean", 0)
        ndwi = spectral.get("ndwi_mean", 0)
        ndbi = spectral.get("ndbi_mean", 0)

        # Index values as metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "NDVI (Vegetation)",
                f"{ndvi:+.3f}",
                delta=f"{spectral.get('vegetation_fraction', 0)*100:.1f}% coverage",
            )
        with c2:
            st.metric(
                "NDWI (Water)",
                f"{ndwi:+.3f}",
                delta=f"{spectral.get('water_fraction', 0)*100:.1f}% coverage",
            )
        with c3:
            st.metric(
                "NDBI (Built-up)",
                f"{ndbi:+.3f}",
                delta=f"{spectral.get('built_up_fraction', 0)*100:.1f}% coverage",
            )

        # Scene composition progress bars
        veg_f = spectral.get("vegetation_fraction", 0)
        wat_f = spectral.get("water_fraction", 0)
        blt_f = spectral.get("built_up_fraction", 0)

        st.markdown("**Scene Composition:**")
        st.progress(min(veg_f, 1.0), text=f"Vegetation: {veg_f*100:.1f}%")
        st.progress(min(wat_f, 1.0), text=f"Water: {wat_f*100:.1f}%")
        st.progress(min(blt_f, 1.0), text=f"Built-up: {blt_f*100:.1f}%")


# ---------------------------------------------------------------------------
# Land Cover Classification Table
# ---------------------------------------------------------------------------


def render_land_cover_table(result: dict[str, Any]) -> None:
    """Render the top-K land cover classification breakdown."""
    vf = result.get("verified_facts", {})
    if not isinstance(vf, dict):
        return
    top_k = vf.get("top_k", [])
    if not top_k:
        return

    with st.expander("Land Cover Classification", expanded=True):
        st.markdown("**Multi-label classification (BigEarthNet-S2 / ResNet-18):**")
        for entry in top_k:
            name = entry.get("class_name", "Unknown")
            prob = entry.get("probability", 0)
            pct = prob * 100
            st.progress(min(prob, 1.0), text=f"{name}: {pct:.1f}%")


# ---------------------------------------------------------------------------
# Detailed Analysis Report
# ---------------------------------------------------------------------------


def render_detailed_report(result: dict[str, Any]) -> None:
    """Render the full markdown analysis report."""
    report_md = result.get("report_markdown", "")
    if report_md:
        st.markdown(report_md, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Minimal Single Consolidated Details Expander (Task 1)
# ---------------------------------------------------------------------------


def render_details_expander(result: dict[str, Any], metas: list[dict] | None = None) -> None:
    """Single collapsed Details expander holding all secondary technical depth."""
    with st.expander("Details & Technical Evidence", expanded=False):
        tab1, tab2, tab3 = st.tabs([
            "📊 Spectral & Classification",
            "📑 Full Report & Export",
            "⚙️ Execution Trace",
        ])

        with tab1:
            # Metadata summary
            if metas:
                st.markdown("**Image Metadata:**")
                for idx, m in enumerate(metas, 1):
                    fn = m.get("filename", f"Image #{idx}")
                    crs = m.get("crs", "N/A")
                    shape = m.get("shape", f"{m.get('width', 'N/A')}x{m.get('height', 'N/A')}")
                    bands = m.get("band_count", m.get("channels", "N/A"))
                    sensor = m.get("sensor", m.get("modality", "N/A"))
                    st.markdown(
                        f"<div class='metadata-mono'>• <strong>{fn}</strong> — "
                        f"Sensor: {sensor} | Shape: {shape} | Bands: {bands} | CRS: {crs}</div>",
                        unsafe_allow_html=True,
                    )
                st.write("")

            # Spectral summary if available
            vf = result.get("verified_facts", {})
            if isinstance(vf, dict) and vf.get("spectral_summary"):
                spectral = vf["spectral_summary"]
                ndvi = spectral.get("ndvi_mean", 0)
                ndwi = spectral.get("ndwi_mean", 0)
                ndbi = spectral.get("ndbi_mean", 0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("NDVI (Vegetation)", f"{ndvi:+.3f}")
                with c2:
                    st.metric("NDWI (Water)", f"{ndwi:+.3f}")
                with c3:
                    st.metric("NDBI (Built-up)", f"{ndbi:+.3f}")

                veg_f = spectral.get("vegetation_fraction", 0)
                wat_f = spectral.get("water_fraction", 0)
                blt_f = spectral.get("built_up_fraction", 0)
                st.progress(min(veg_f, 1.0), text=f"Vegetation: {veg_f*100:.1f}%")
                st.progress(min(wat_f, 1.0), text=f"Water: {wat_f*100:.1f}%")
                st.progress(min(blt_f, 1.0), text=f"Built-up: {blt_f*100:.1f}%")
                st.write("")

            # Land cover top-k
            top_k = vf.get("top_k", []) if isinstance(vf, dict) else []
            if top_k:
                st.markdown("**Land Cover Surface Classes (ResNet-18 / BigEarthNet):**")
                for entry in top_k:
                    name = entry.get("class_name", "Unknown")
                    prob = entry.get("probability", 0)
                    st.progress(min(prob, 1.0), text=f"{name}: {prob*100:.1f}%")

            if not (isinstance(vf, dict) and (vf.get("spectral_summary") or vf.get("top_k"))):
                st.info("No Earth-observation spectral indices or land-cover distribution for this analysis.")

        with tab2:
            render_detailed_report(result)
            st.divider()
            render_download_buttons(result)

        with tab3:
            trace = result.get("trace", {})
            if trace:
                exec_time = trace.get("execution_time_seconds")
                t_str = f"{exec_time:.2f}s" if exec_time is not None else "N/A"
                st.markdown(
                    f"<div class='metadata-mono'>"
                    f"<strong>Task:</strong> <code>{trace.get('task', 'N/A')}</code> | "
                    f"<strong>Duration:</strong> <code>{t_str}</code> | "
                    f"<strong>Confidence:</strong> <code>{trace.get('confidence', 'N/A')}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                tools = trace.get("tools_invoked", [])
                if tools:
                    st.markdown(
                        "<div class='metadata-mono'><strong>Pipeline tools:</strong> " +
                        " &rarr; ".join(f"<code>{t}</code>" for t in tools) + "</div>",
                        unsafe_allow_html=True,
                    )
                st.write("")
                st.json(trace)
            else:
                st.info("No execution trace recorded.")


# ---------------------------------------------------------------------------
# Batch Queue Item Component (Task 2)
# ---------------------------------------------------------------------------


def render_batch_item_card(idx: int, item: dict[str, Any]) -> None:
    """Render a single stacked summary card for a batch item."""
    status = item.get("status", "queued")
    query = item.get("query", "")
    res = item.get("result")
    err = item.get("error")

    status_cls = f"status-{status}"
    st.markdown(
        f"<div class='batch-card'>"
        f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
        f"<strong>Item #{idx + 1}</strong>"
        f"<span class='batch-status-badge {status_cls}'>{status.upper()}</span>"
        f"</div>"
        f"<div style='font-size: 0.9rem; color: #94a3b8; margin-top: 4px;'>"
        f"Query: <em>{query[:80]}{'...' if len(query) > 80 else ''}</em>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if status == "error" and err:
        st.error(f"Item #{idx + 1} failed: {err}")
    elif status == "done" and res:
        with st.expander(f"View Results for Item #{idx + 1}", expanded=False):
            render_confidence_badge(res)
            st.markdown(res.get("answer", ""))
            render_overlay(res)
            render_details_expander(res, item.get("metas"))

