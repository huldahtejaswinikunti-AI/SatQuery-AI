"""SatQuery AI -- Streamlit entry point.

Launch with:  streamlit run app/main.py

Upload flow supports three configurations:
  1. One image (single-image VQA / captioning / grounding)
  2. Two co-registered images of different modalities (optical + SAR fusion)
  3. Two same-location images from different dates (change detection)

Every model call has a timeout.  Errors are shown as clear messages, never
raw stack traces.  Optional result fields degrade gracefully to "not shown".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import numpy as np
import streamlit as st

# ---- Page config (must be first Streamlit call) -------------------------
st.set_page_config(
    page_title="SatQuery AI",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Internal imports ---------------------------------------------------
try:
    from app.session_state import (
        init_session_state,
        reset_session_state,
        store_result,
        store_error,
    )
    from app.ui_components import (
        render_header,
        render_image_preview,
        render_confidence_badge,
        render_consensus_score,
        render_change_direction,
        render_overlay,
        render_execution_trace,
        render_download_buttons,
        render_spectral_summary,
        render_land_cover_table,
        render_detailed_report,
    )
    from app.pipeline_bridge import run_pipeline
except ImportError:
    from session_state import (
        init_session_state,
        reset_session_state,
        store_result,
        store_error,
    )
    from ui_components import (
        render_header,
        render_image_preview,
        render_confidence_badge,
        render_consensus_score,
        render_change_direction,
        render_overlay,
        render_execution_trace,
        render_download_buttons,
        render_spectral_summary,
        render_land_cover_table,
        render_detailed_report,
    )
    from pipeline_bridge import run_pipeline

from satquery.utils.geo_io import load_image_as_array

# ---- Load CSS -----------------------------------------------------------
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---- Session state ------------------------------------------------------
init_session_state()

# ---- Load demo queries --------------------------------------------------
DEMO_QUERIES_PATH = Path(__file__).resolve().parent.parent / "demo" / "demo_queries.json"

@st.cache_data
def _load_demo_queries():
    if DEMO_QUERIES_PATH.exists():
        with open(DEMO_QUERIES_PATH) as f:
            return json.load(f)
    return []

demo_queries = _load_demo_queries()

# ---- Header -------------------------------------------------------------
render_header()

# ---- Sidebar: input configuration --------------------------------------
with st.sidebar:
    st.header("1. Input Configuration")

    # Demo preset dropdown
    preset_labels = ["-- Free-text query --"] + [
        dq.get("label", dq.get("query", "")[:60]) for dq in demo_queries
    ]
    preset_idx = st.selectbox(
        "Load Curated Demo Scenario:",
        range(len(preset_labels)),
        format_func=lambda i: preset_labels[i],
        index=0,
        key="sidebar_preset",
    )

    st.divider()

    # Upload
    st.markdown(
        "**Upload Imagery** (1 or 2 files)  \n"
        "Formats: GeoTIFF / TIFF for real data; PNG / JPEG for benchmark samples."
    )
    uploaded = st.file_uploader(
        "Upload satellite images",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and len(uploaded) > 2:
        st.error(
            "At most 2 images are supported.  Upload a single image or a pair "
            "(optical + SAR, or bi-temporal)."
        )
        uploaded = uploaded[:2]

    st.divider()

    if st.button("Reset / Clear", use_container_width=True):
        reset_session_state()
        st.rerun()


# ---- Process demo preset or upload --------------------------------------

# Apply demo preset if selected
if preset_idx > 0:
    dq = demo_queries[preset_idx - 1]
    default_query = dq.get("query", "")

    # Load demo images from filenames
    demo_base = Path(__file__).resolve().parent.parent / "data" / "demo_samples"
    demo_files = dq.get("sample_files", []) or dq.get("files", [])
    if not demo_files and dq.get("file"):
        demo_files = [dq["file"]]

    if demo_files:
        imgs, ms = [], []
        for fname in demo_files:
            # Check direct relative path first
            p_direct = demo_base / fname
            if p_direct.exists():
                arr, meta = load_image_as_array(str(p_direct))
                imgs.append(arr)
                ms.append(meta)
                continue
            # Search in subdirectories by filename
            base_name = Path(fname).name
            for subdir in ["single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs"]:
                p = demo_base / subdir / base_name
                if p.exists():
                    arr, meta = load_image_as_array(str(p))
                    imgs.append(arr)
                    ms.append(meta)
                    break
        if imgs:
            st.session_state.images = imgs
            st.session_state.metas = ms
    else:
        # Generate synthetic demo if no files specified
        h, w = 256, 256
        default_img = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)
        st.session_state.images = [default_img]
        st.session_state.metas = [{"filename": "synthetic_demo.png", "format": "PNG",
                                    "band_count": 3, "width": w, "height": h}]

else:
    default_query = "Describe the land-cover and features visible in this satellite imagery."

# Process uploaded files
if uploaded:
    imgs, ms = [], []
    for f in uploaded:
        arr, meta = load_image_as_array(f)
        imgs.append(arr)
        ms.append(meta)
    st.session_state.images = imgs
    st.session_state.metas = ms

# ---- Main layout --------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    render_image_preview(st.session_state.images, st.session_state.metas)

    query = st.text_area(
        "Natural Language Query:",
        value=default_query,
        height=85,
        key="query_area",
    )

    run_btn = st.button(
        "Analyze & Ground",
        type="primary",
        use_container_width=True,
    )

with col_right:
    st.subheader("Analysis & Evidence Grounding")

    # ---- Run pipeline on button click -----------------------------------
    if run_btn:
        if not st.session_state.images:
            st.error(
                "Please provide at least one satellite image.  Upload your "
                "own imagery or select a demo scenario from the sidebar."
            )
        elif not query.strip():
            st.error("Please enter a query about the image(s).")
        else:
            with st.spinner(
                "Executing pipeline with RS cross-verification... "
                "(this may take 10-30 seconds on GPU)"
            ):
                result = run_pipeline(
                    st.session_state.images,
                    st.session_state.metas,
                    query,
                )

                if result.get("confidence_tag") == "error":
                    # Show validation/pipeline error clearly
                    reason = result.get("validation_failure_reason", "")
                    if reason:
                        st.error(f"**Analysis could not proceed:** {reason}")
                    else:
                        st.error(f"**Analysis failed:** {result.get('answer', 'Unknown error')}")
                else:
                    store_result(result)

    # ---- Render result --------------------------------------------------
    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result

        # Confidence badge (primary)
        render_confidence_badge(res)

        # Consensus score headline (if present)
        render_consensus_score(res)

        # Answer text
        st.markdown(res.get("answer", ""))

        # Spectral analysis & scene composition
        render_spectral_summary(res)

        # Land cover classification breakdown
        render_land_cover_table(res)

        # Change direction (if present)
        render_change_direction(res)

        # Overlay image (if present)
        render_overlay(res)

        # Detailed analysis report
        render_detailed_report(res)

        # Execution trace
        render_execution_trace(res)

        # Download buttons
        render_download_buttons(res)

    elif st.session_state.error_message:
        st.error(st.session_state.error_message)

    else:
        # Empty state -- helpful prompt
        st.markdown(
            """
            **Getting started:**

            1. Select a **demo scenario** from the sidebar, or upload your own satellite imagery
            2. Enter a natural-language **query** about the image
            3. Click **Analyze & Ground** to run the full pipeline

            **Supported tasks:**
            - Single-image VQA and captioning
            - Open-vocabulary grounding (water bodies, built-up areas, vegetation)
            - Bi-temporal change detection
            - Optical + SAR fusion for cloud-penetrating analysis

            **Example queries:**
            - *"Describe the land-cover and major objects visible in this image."*
            - *"Highlight the water body referred to in the query."*
            - *"What changed between these two dates?"*
            - *"Use optical and SAR together to identify built-up structures under cloud."*
            """
        )
