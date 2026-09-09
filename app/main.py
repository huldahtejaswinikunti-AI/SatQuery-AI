"""SatQuery AI -- Streamlit main entry point.

Redesigned minimal, focused interface supporting:
- Earth Observation Analysis (Sentinel-1/2, Land Cover, Optical+SAR Fusion, Change Detection)
- Chandrayaan-2 Lunar Analysis (Zero-shot VQA, Crater Morphology, PSR Shadow Analysis)
- Multi-Input / Batch Queuing with Sequential Execution & Error Isolation
- Minimal clutter: Answer, Confidence Badge, and Overlay on top; deep technical metrics
  tucked neatly into a single collapsed "Details" expander.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_APP = Path(__file__).resolve().parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import numpy as np
import streamlit as st

# ---- Page config --------------------------------------------------------
st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
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
        render_overlay,
        render_details_expander,
        render_batch_item_card,
    )
    from app.pipeline_bridge import (
        run_pipeline,
        run_lunar_pipeline,
        run_batch_pipeline,
    )
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
        render_overlay,
        render_details_expander,
        render_batch_item_card,
    )
    from pipeline_bridge import (
        run_pipeline,
        run_lunar_pipeline,
        run_batch_pipeline,
    )

from satquery.utils.geo_io import load_image_as_array

# ---- Load Custom CSS ----------------------------------------------------
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---- Session state ------------------------------------------------------
init_session_state()

# ---- Load Datasets & Demo Presets ---------------------------------------
DEMO_BASE = Path(__file__).resolve().parent.parent / "data" / "demo_samples"
EARTH_QUERIES_PATH = Path(__file__).resolve().parent.parent / "demo" / "demo_queries.json"
LUNAR_QUERIES_PATH = Path(__file__).resolve().parent.parent / "demo" / "lunar_demo_queries.json"


@st.cache_data
def _load_json_catalog(path: Path) -> list[dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


earth_presets = _load_json_catalog(EARTH_QUERIES_PATH)
lunar_presets = _load_json_catalog(LUNAR_QUERIES_PATH)

# ---- Top-level Header ---------------------------------------------------
render_header()

# ---- Task 1 & 3: Obvious Top-level Mode Selector ------------------------
top_col1, top_col2, top_col3 = st.columns([2, 1, 1])

with top_col1:
    mode = st.radio(
        "Observation Domain:",
        options=["Earth analysis", "Lunar analysis"],
        horizontal=True,
        key="top_mode_select",
        help="Select between Earth land-cover cross-verification and Chandrayaan-2 lunar exploration.",
    )
    st.session_state.analysis_mode = mode

with top_col3:
    flow_type = st.radio(
        "Workflow:",
        options=["Single Analysis", "Batch Queue"],
        horizontal=True,
        key="flow_type_select",
    )

st.divider()

# Helper to load demo image files
def _resolve_image_files(filenames: list[str], is_lunar: bool = False) -> tuple[list[np.ndarray], list[dict]]:
    imgs, metas = [], []
    base = DEMO_BASE / "lunar" if is_lunar else DEMO_BASE
    for fname in filenames:
        p_direct = base / fname
        if p_direct.exists():
            arr, meta = load_image_as_array(str(p_direct))
            imgs.append(arr)
            metas.append(meta)
            continue
        # Fallback search across subdirectories
        base_name = Path(fname).name
        for subdir in ["", "single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs", "lunar"]:
            candidate = DEMO_BASE / subdir / base_name
            if candidate.exists():
                arr, meta = load_image_as_array(str(candidate))
                imgs.append(arr)
                metas.append(meta)
                break
    return imgs, metas


# ---- Picture Selection & Input Configuration ----------------------------
current_images = []
current_metas = []
default_query = ""

if mode == "Earth analysis":
    st.markdown("### Select Earth Observation Imagery")
    earth_tabs = ["Curated Scenarios", "Custom Upload"]
    chosen_tab = st.radio("Input Source:", earth_tabs, horizontal=True, label_visibility="collapsed")

    if chosen_tab == "Curated Scenarios":
        # Group presets by type to give options across modes
        scenario_labels = [f"{i+1}. {p.get('label', p.get('query', '')[:50])}" for i, p in enumerate(earth_presets)]
        scenario_idx = st.selectbox(
            "Select Curated Demo Scene:",
            range(len(scenario_labels)),
            format_func=lambda i: scenario_labels[i],
            index=0,
            key="earth_preset_select",
        )
        selected_preset = earth_presets[scenario_idx]
        default_query = selected_preset.get("query", "")
        f_list = selected_preset.get("sample_files") or selected_preset.get("files") or []
        current_images, current_metas = _resolve_image_files(f_list, is_lunar=False)
        st.caption(f"**Why this matters to ISRO:** {selected_preset.get('why_this_matters_to_isro', 'Standard remote sensing workflow.')}")

    else:
        st.markdown(
            "**Upload Imagery** (1 single image, or 2 paired images for Optical+SAR / Bi-Temporal Change Detection).  \n"
            "*Supported formats: GeoTIFF / TIFF / PNG / JPEG*"
        )
        uploaded = st.file_uploader(
            "Upload Earth Images",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="earth_uploader",
        )
        if uploaded:
            for f in uploaded[:2]:
                arr, meta = load_image_as_array(f)
                current_images.append(arr)
                current_metas.append(meta)
        default_query = "Describe the land-cover surface features and assess terrain composition."

else:
    # Lunar Analysis Mode
    st.markdown("### Chandrayaan-2 Lunar Surface Imagery (OHRC / TMC-2)")
    lunar_tabs = ["Chandrayaan-2 Demo Targets", "Custom Lunar Upload"]
    chosen_lunar_tab = st.radio("Input Source:", lunar_tabs, horizontal=True, label_visibility="collapsed")

    if chosen_lunar_tab == "Chandrayaan-2 Demo Targets":
        lunar_labels = [f"{i+1}. {p.get('label', p.get('query', '')[:50])}" for i, p in enumerate(lunar_presets)]
        lunar_idx = st.selectbox(
            "Select Lunar Target Feature:",
            range(len(lunar_labels)),
            format_func=lambda i: lunar_labels[i],
            index=0,
            key="lunar_preset_select",
        )
        selected_lunar = lunar_presets[lunar_idx]
        default_query = selected_lunar.get("query", "")
        l_files = [selected_lunar["file"]] if "file" in selected_lunar else []
        current_images, current_metas = _resolve_image_files(l_files, is_lunar=True)
        st.caption(f"**ISRO / ISSDC Context:** {selected_lunar.get('why_it_matters', 'Chandrayaan-2 mission science exploration.')}")

    else:
        st.markdown(
            "**Upload Lunar Raster Product** (Chandrayaan-2 OHRC/TMC-2 GeoTIFF, TIFF, or PNG)."
        )
        lunar_uploaded = st.file_uploader(
            "Upload Lunar Product",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="lunar_uploader",
        )
        if lunar_uploaded:
            arr, meta = load_image_as_array(lunar_uploaded)
            current_images = [arr]
            current_metas = [meta]
        default_query = "Identify prominent impact craters, ejecta deposits, and shadowed regions."

# Update active images in session state
st.session_state.images = current_images
st.session_state.metas = current_metas

st.write("")

# ---- Batch Queue Mode (Task 2) ------------------------------------------
if flow_type == "Batch Queue":
    st.subheader("Batch Queue Management")
    st.markdown(
        "Queue several independent valid input-groups to process sequentially. "
        "Each item runs with isolated error handling so one invalid upload will never block the rest."
    )

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        batch_query = st.text_input(
            "Query for this item (or leave blank to use default):",
            value=default_query,
            key="batch_item_query_input",
        )
    with col_q2:
        st.write("")
        st.write("")
        if st.button("➕ Add Current Selection to Queue", use_container_width=True):
            if not st.session_state.images:
                st.warning("Please select or upload image(s) before adding to batch.")
            else:
                st.session_state.batch_queue.append({
                    "images": list(st.session_state.images),
                    "metas": list(st.session_state.metas),
                    "query": batch_query.strip() or default_query,
                    "status": "queued",
                    "result": None,
                    "error": None,
                })
                st.success(f"Added item #{len(st.session_state.batch_queue)} to batch queue.")
                st.rerun()

    # Show current queue status
    queue_len = len(st.session_state.batch_queue)
    st.markdown(f"**Queued Items ({queue_len}):**")

    if queue_len == 0:
        st.info("The batch queue is currently empty. Use 'Add Current Selection to Queue' above to stage items.")
    else:
        for q_idx, it in enumerate(st.session_state.batch_queue):
            render_batch_item_card(q_idx, it)

        c_run1, c_run2 = st.columns([2, 1])
        with c_run1:
            run_batch_btn = st.button(
                f"🚀 Process Entire Batch ({queue_len} items)",
                type="primary",
                use_container_width=True,
            )
        with c_run2:
            if st.button("🗑️ Clear Batch Queue", use_container_width=True):
                st.session_state.batch_queue = []
                st.rerun()

        if run_batch_btn:
            progress_bar = st.progress(0, text="Starting sequential batch run...")
            status_text = st.empty()

            def _batch_progress(idx, total, status, res):
                frac = (idx + 1) / total
                progress_bar.progress(frac, text=f"Processing item {idx + 1} of {total} ({status})...")
                status_text.caption(f"Currently processing item #{idx + 1} ({status})")

            current_mode = "lunar" if mode == "Lunar analysis" else "earth"
            with st.spinner("Processing batch queue sequentially..."):
                updated_queue = run_batch_pipeline(
                    st.session_state.batch_queue,
                    mode=current_mode,
                    progress_callback=_batch_progress,
                )
                st.session_state.batch_queue = updated_queue

            progress_bar.progress(1.0, text="Batch processing complete!")
            st.success(f"Processed {queue_len} batch items. Results are displayed above.")
            st.rerun()

# ---- Single Analysis Minimal Flow (Task 1 & Task 3) --------------------
else:
    # Single clear vertical flow: Image Preview -> Query -> Run -> Result -> Collapsed Details
    col_preview, col_query = st.columns([1, 1], gap="medium")

    with col_preview:
        render_image_preview(st.session_state.images, st.session_state.metas)

    with col_query:
        st.markdown("**Natural Language Query:**")
        query_text = st.text_area(
            "Query:",
            value=default_query,
            height=90,
            label_visibility="collapsed",
            key="single_query_input",
        )

        run_single_btn = st.button(
            "🚀 Run SatQuery AI Analysis",
            type="primary",
            use_container_width=True,
        )

        if run_single_btn:
            if not st.session_state.images:
                st.error("Please provide at least one satellite or lunar image before running analysis.")
            elif not query_text.strip():
                st.error("Please enter a query about the image.")
            else:
                with st.spinner("Executing analysis pipeline..."):
                    if mode == "Lunar analysis":
                        res = run_lunar_pipeline(
                            st.session_state.images,
                            st.session_state.metas,
                            query_text,
                        )
                    else:
                        res = run_pipeline(
                            st.session_state.images,
                            st.session_state.metas,
                            query_text,
                        )

                    if res.get("confidence_tag") == "error":
                        reason = res.get("validation_failure_reason", "") or res.get("answer", "Unknown error")
                        st.error(f"Analysis could not proceed: {reason}")
                        store_error(reason)
                    else:
                        store_result(res)

    # ---- Result Section (Minimal, zero clutter) -------------------------
    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result
        st.divider()
        st.markdown("### Analysis Result")

        # 1. Confidence Badge
        render_confidence_badge(res)

        # 2. Plain Language Answer
        st.markdown(res.get("answer", ""))

        # 3. Visual Overlay (if present)
        render_overlay(res)

        # 4. Single Collapsed Details Expander (Task 1: All secondary depth in ONE expander)
        render_details_expander(res, st.session_state.metas)

    elif st.session_state.error_message:
        st.error(st.session_state.error_message)

    else:
        st.info("Select an image preset or upload imagery above, enter your question, and click **Run SatQuery AI Analysis**.")
