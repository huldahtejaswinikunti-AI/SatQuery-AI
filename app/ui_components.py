from __future__ import annotations
import json
from typing import Any
import numpy as np
import streamlit as st
from satquery.pipeline.report_generator import ReportGenerator
from satquery.utils.image_utils import to_display_rgb

def render_header() -> None:
    st.markdown('''
        <div style="padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
            <h1 style="margin-bottom: 2px; color: #1E88E5; font-size: 2.2rem;">??? SatQuery AI</h1>
            <p style="color: #9E9E9E; font-size: 1.05rem; margin-top: 0;">
                Agentic Vision-Language Assistant for Remote Sensing with Classical Signal Grounding
                <br>
                <span style="font-size: 0.85rem; color: #B0BEC5;">SIH 2026 ? Problem Statement 26167 ? ISRO/SAC</span>
            </p>
        </div>
    ''', unsafe_allow_html=True)

def render_confidence_badge(tag: str, score: float) -> None:
    pct = int(round(score * 100))
    if tag in ("high_cross_verified", "high_sar_penetration"):
        cls, lbl = "badge-high-verified", f"??? High Confidence (Cross-Verified: {pct}%)"
    elif tag == "high_rule_based":
        cls, lbl = "badge-high-rule", f"?? Deterministic Signal ({pct}%)"
    elif tag == "lower_confidence_disagreement":
        cls, lbl = "badge-disagreement", f"?? Lower Confidence (Signal Disagreement: {pct}%)"
    elif tag == "error":
        cls, lbl = "badge-disagreement", "? Input Validation Error"
    else:
        cls, lbl = "badge-unverified", f"?? Moderate Confidence (Unverified: {pct}%)"
    st.markdown(f'<div class="badge-container {cls}">{lbl}</div>', unsafe_allow_html=True)

def render_image_preview(images: list[np.ndarray], metas: list[dict]) -> None:
    if not images:
        st.info("Upload 1 or 2 satellite images or pick a pre-loaded demo scenario.")
        return
    st.subheader("Input Imagery")
    if len(images) == 1:
        rgb = to_display_rgb(images[0])
        name = metas[0].get("filename", "Acquisition 1") if metas else "Acquisition 1"
        st.image(rgb, caption=f"{name} ({images[0].shape[1]}x{images[0].shape[0]})", use_container_width=True)
    else:
        cols = st.columns(2)
        for i, (img, meta) in enumerate(zip(images, metas)):
            with cols[i]:
                rgb = to_display_rgb(img)
                name = meta.get("filename", f"Image {i+1}")
                st.image(rgb, caption=f"{name} ({img.shape[1]}x{img.shape[0]})", use_container_width=True)

def render_execution_trace(trace: dict[str, Any]) -> None:
    with st.expander("?? Auditable Execution Trace & Tool Timeline", expanded=False):
        st.markdown(f"**Trace ID:** `{trace.get('trace_id', 'N/A')}` | **Task:** `{trace.get('task', 'N/A')}`")
        st.markdown(f"**Tools Invoked:** `{', '.join(trace.get('tools_invoked', []))}`")
        for step in trace.get("steps", []):
            st.markdown(f'''
                <div class="trace-step">
                    <span class="trace-step-name">?? {step.get('tool_name')}</span>
                    <span class="trace-duration">?? {step.get('duration_ms', 0):.1f} ms</span>
                </div>
            ''', unsafe_allow_html=True)
            if step.get("output_summary"):
                st.caption(f"Outputs: {json.dumps(step['output_summary'])}")
        st.json(trace)

def render_download_buttons(result: dict[str, Any]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("?? Download Markdown Report", ReportGenerator.generate_markdown(result), "satquery_report.md", "text/markdown", use_container_width=True)
    with col2:
        st.download_button("?? Download Full JSON Trace", ReportGenerator.generate_json(result), "satquery_trace.json", "application/json", use_container_width=True)
