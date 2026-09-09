"""SatQuery AI -- Streamlit main entry point.

Redesigned minimal, high-contrast interface with:
- Top-level Domain Navigation (Earth analysis vs Lunar analysis)
- Multiple Curated Picture Options per mode
- Visual grouping using bordered card containers (no black-on-black floating elements)
- Balanced layout without empty gray voids
- High-contrast primary call-to-action buttons (#2dd4bf filled teal)
- Clean result section with single collapsed Details expander
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

# Helper to resolve demo images
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
        base_name = Path(fname).name
        for subdir in ["", "single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs", "lunar"]:
            candidate = DEMO_BASE / subdir / base_name
            if candidate.exists():
                arr, meta = load_image_as_array(str(candidate))
                imgs.append(arr)
                metas.append(meta)
                break
    return imgs, metas


# ---- Header -------------------------------------------------------------
render_header()

# ---- Container 1: Domain Navigation & Workflow Control ------------------
with st.container(border=True):
    col_nav1, col_nav2 = st.columns([3, 2])
    with col_nav1:
        mode = st.radio(
            "Observation Domain:",
            options=["Earth analysis", "Lunar analysis"],
            horizontal=True,
            key="domain_select",
            help="Switch between Earth land-cover cross-verification and Chandrayaan-2 lunar exploration.",
        )
        st.session_state.analysis_mode = mode
    with col_nav2:
        flow_type = st.radio(
            "Workflow:",
            options=["Single Analysis", "Batch Queue"],
            horizontal=True,
            key="workflow_select",
        )

# ---- Container 2: Scenario & Picture Options Picker ---------------------
current_images = []
current_metas = []
default_query = ""

with st.container(border=True):
    if mode == "Earth analysis":
        st.markdown("#### 🌍 Earth Observation Scenario & Picture Selection")
        earth_tab_names = ["Curated Scenarios", "Custom Image Upload"]
        chosen_earth_source = st.radio(
            "Source Mode:",
            earth_tab_names,
            horizontal=True,
            label_visibility="collapsed",
            key="earth_source_radio",
        )

        if chosen_earth_source == "Curated Scenarios":
            # Multiple options presentation with rich context
            options_labels = [
                f"{i+1}. {p.get('label', p.get('query', '')[:50])}"
                for i, p in enumerate(earth_presets)
            ]
            selected_idx = st.selectbox(
                "Choose Picture Scenario to Analyze:",
                range(len(options_labels)),
                format_func=lambda i: options_labels[i],
                index=0,
                key="earth_scenario_dropdown",
            )
            selected_preset = earth_presets[selected_idx]
            default_query = selected_preset.get("query", "")
            f_list = selected_preset.get("sample_files") or selected_preset.get("files") or []
            current_images, current_metas = _resolve_image_files(f_list, is_lunar=False)

            # Auto-update query box when preset changes
            if st.session_state.get("last_earth_preset_idx") != selected_idx:
                st.session_state["last_earth_preset_idx"] = selected_idx
                st.session_state["main_query_text_area"] = default_query

            task_type = selected_preset.get("task", "Remote Sensing Analysis")
            why_isro = selected_preset.get("why_this_matters_to_isro", "Operational satellite evaluation.")
            st.caption(f"**Task Type:** `{task_type}` | **ISRO/SAC Context:** {why_isro}")

        else:
            st.markdown(
                "Upload 1 single optical/SAR image, or 2 paired images for **Optical+SAR Fusion** or **Change Detection**.  \n"
                "*Supported formats: GeoTIFF, TIFF, PNG, JPEG*"
            )
            uploaded_files = st.file_uploader(
                "Upload Earth Images",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="earth_custom_uploader",
            )
            if uploaded_files:
                for f in uploaded_files[:2]:
                    arr, meta = load_image_as_array(f)
                    current_images.append(arr)
                    current_metas.append(meta)
            default_query = "Describe the land-cover surface features and assess terrain composition."

    else:
        st.markdown("#### 🌕 Chandrayaan-2 Lunar Surface Imagery (OHRC / TMC-2)")
        lunar_tab_names = ["Curated Chandrayaan-2 Targets", "Custom Lunar Product Upload"]
        chosen_lunar_source = st.radio(
            "Lunar Source:",
            lunar_tab_names,
            horizontal=True,
            label_visibility="collapsed",
            key="lunar_source_radio",
        )

        if chosen_lunar_source == "Curated Chandrayaan-2 Targets":
            lunar_options = [
                f"{i+1}. {p.get('label', p.get('query', '')[:50])}"
                for i, p in enumerate(lunar_presets)
            ]
            selected_lunar_idx = st.selectbox(
                "Choose Chandrayaan-2 Picture Option:",
                range(len(lunar_options)),
                format_func=lambda i: lunar_options[i],
                index=0,
                key="lunar_scenario_dropdown",
            )
            selected_lunar = lunar_presets[selected_lunar_idx]
            default_query = selected_lunar.get("query", "")
            l_files = [selected_lunar["file"]] if "file" in selected_lunar else []
            current_images, current_metas = _resolve_image_files(l_files, is_lunar=True)

            # Auto-update query box when lunar preset changes
            if st.session_state.get("last_lunar_preset_idx") != selected_lunar_idx:
                st.session_state["last_lunar_preset_idx"] = selected_lunar_idx
                st.session_state["main_query_text_area"] = default_query

            st.caption(f"**ISRO / ISSDC Context:** {selected_lunar.get('why_it_matters', 'Lunar exploration.')}")

        else:
            st.markdown(
                "Upload calibrated Chandrayaan-2 OHRC/TMC-2 raster product (.tif, .png, or converted .IMG)."
            )
            lunar_upload = st.file_uploader(
                "Upload Lunar Raster",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                accept_multiple_files=False,
                label_visibility="collapsed",
                key="lunar_custom_uploader",
            )
            if lunar_upload:
                arr, meta = load_image_as_array(lunar_upload)
                current_images = [arr]
                current_metas = [meta]
            default_query = "Identify prominent impact craters, ejecta deposits, and shadowed regions."

# Sync active images in session state
st.session_state.images = current_images
st.session_state.metas = current_metas

# ---- Batch Queue Mode ---------------------------------------------------
if flow_type == "Batch Queue":
    with st.container(border=True):
        st.markdown("#### 📋 Sequential Batch Queue")
        st.markdown(
            "Stage multiple independent inputs to process in sequence with per-item error isolation."
        )

        col_bq1, col_bq2 = st.columns([3, 1])
        with col_bq1:
            batch_item_query = st.text_input(
                "Query for staged item (or leave default):",
                value=default_query,
                key="batch_staged_query",
            )
        with col_bq2:
            st.write("")
            st.write("")
            if st.button("➕ Stage to Queue", use_container_width=True):
                if not st.session_state.images:
                    st.warning("Select or upload image(s) before staging to queue.")
                else:
                    st.session_state.batch_queue.append({
                        "images": list(st.session_state.images),
                        "metas": list(st.session_state.metas),
                        "query": batch_item_query.strip() or default_query,
                        "status": "queued",
                        "result": None,
                        "error": None,
                    })
                    st.success(f"Staged Item #{len(st.session_state.batch_queue)} to batch queue.")
                    st.rerun()

        queue_len = len(st.session_state.batch_queue)
        st.markdown(f"**Queued Items ({queue_len}):**")

        if queue_len == 0:
            st.info("The queue is empty. Click 'Stage to Queue' above to add items.")
        else:
            for q_idx, item in enumerate(st.session_state.batch_queue):
                render_batch_item_card(q_idx, item)

            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                run_all_batch = st.button(
                    f"🚀 Execute Batch Queue ({queue_len} items)",
                    type="primary",
                    use_container_width=True,
                    key="run_all_batch_btn",
                )
            with col_btn2:
                if st.button("🗑️ Clear Queue", use_container_width=True, key="clear_batch_btn"):
                    st.session_state.batch_queue = []
                    st.rerun()

            if run_all_batch:
                p_bar = st.progress(0, text="Executing batch queue...")
                active_mode = "lunar" if mode == "Lunar analysis" else "earth"

                def _prog_update(idx, total, status, res):
                    frac = (idx + 1) / total
                    p_bar.progress(frac, text=f"Processing item {idx + 1} of {total} ({status})...")

                with st.spinner("Processing batch items sequentially..."):
                    updated = run_batch_pipeline(
                        st.session_state.batch_queue,
                        mode=active_mode,
                        progress_callback=_prog_update,
                    )
                    st.session_state.batch_queue = updated

                p_bar.progress(1.0, text="Batch execution finished!")
                st.success(f"Processed {queue_len} items. Results are ready above.")
                st.rerun()

# ---- Single Analysis Layout (Balanced Two-Column Card) ------------------
else:
    with st.container(border=True):
        st.markdown("#### 🖼️ Visual Input & Analysis Query")
        col_img, col_act = st.columns([1, 1], gap="medium")

        with col_img:
            render_image_preview(st.session_state.images, st.session_state.metas)

        with col_act:
            st.markdown("**Natural Language Query:**")
            active_query = st.text_area(
                "Query Prompt:",
                value=default_query,
                height=95,
                label_visibility="collapsed",
                key="main_query_text_area",
            )

            st.write("")
            run_analysis_btn = st.button(
                "🚀 Run SatQuery AI Analysis",
                type="primary",
                use_container_width=True,
                key="run_single_analysis_btn",
            )

            if run_analysis_btn:
                if not st.session_state.images:
                    st.error("Please select or upload at least one image.")
                elif not active_query.strip():
                    st.error("Please enter a question or query.")
                else:
                    with st.spinner("Executing analysis pipeline & cross-verification..."):
                        if mode == "Lunar analysis":
                            result = run_lunar_pipeline(
                                st.session_state.images,
                                st.session_state.metas,
                                active_query,
                            )
                        else:
                            result = run_pipeline(
                                st.session_state.images,
                                st.session_state.metas,
                                active_query,
                            )

                        if result.get("confidence_tag") == "error":
                            err_reason = result.get("validation_failure_reason") or result.get("answer", "Unknown error")
                            st.error(f"Analysis failed: {err_reason}")
                            store_error(err_reason)
                        else:
                            store_result(result)

    # ---- Results Section (Only renders when results exist) --------------
    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result
        with st.container(border=True):
            st.markdown("### 🎯 Analysis Results")

            # 1. Primary Confidence Badge
            render_confidence_badge(res)

            # 2. Grounded Natural-Language Answer
            st.markdown(res.get("answer", "No textual synthesis generated."))

            # 3. Visual Overlay Segmentation / Change Mask
            render_overlay(res)

            # 4. Single Collapsed Details Expander (Zero Clutter)
            render_details_expander(res, st.session_state.metas)

    elif st.session_state.error_message:
        st.error(st.session_state.error_message)
