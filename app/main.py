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
        render_3d_planetary_hero,
        render_image_preview,
        render_image_picker_gallery,
        render_confidence_badge,
        render_overlay,
        render_details_expander,
        render_batch_item_card,
        render_scientific_report_panel,
        render_staging_telemetry_panel,
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
        render_3d_planetary_hero,
        render_image_preview,
        render_image_picker_gallery,
        render_confidence_badge,
        render_overlay,
        render_details_expander,
        render_batch_item_card,
        render_scientific_report_panel,
        render_staging_telemetry_panel,
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
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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


# ---- 3D Planetary Hero Experience (Three.js WebGL Summit) ----------------
render_3d_planetary_hero(st.session_state.get("domain_select", "Earth analysis"))

# Scroll Anchor for Hero CTA ("EXPLORE OBSERVATIONS ↓")
st.markdown('<div id="observation-workspace" style="scroll-margin-top: 15px; margin-bottom: 8px;"></div>', unsafe_allow_html=True)

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

# Helper function to resolve thumbnail array
def _resolve_single_thumb(file_or_path: str) -> np.ndarray | None:
    is_lun = (mode == "Lunar analysis")
    imgs, _ = _resolve_image_files([file_or_path], is_lunar=is_lun)
    return imgs[0] if imgs else None

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
            # Scenario selector dropdown
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

            # Track scenario change to reset selected image index
            if st.session_state.get("last_earth_preset_idx") != selected_idx:
                st.session_state["last_earth_preset_idx"] = selected_idx
                st.session_state["selected_earth_img_idx"] = 0

            # Visual Multiple Image Options Picker
            sample_items = selected_preset.get("sample_items") or []
            selected_img_idx = st.session_state.get("selected_earth_img_idx", 0)

            if sample_items:
                selected_img_idx = render_image_picker_gallery(
                    sample_items=sample_items,
                    selected_idx=selected_img_idx,
                    key_prefix=f"earth_scen_{selected_idx}",
                    resolve_thumb_fn=_resolve_single_thumb,
                )
                st.session_state["selected_earth_img_idx"] = selected_img_idx

                chosen_item = sample_items[selected_img_idx]
                default_query = chosen_item.get("query") or selected_preset.get("query", "")
                f_list = chosen_item.get("files") or ([chosen_item["file"]] if "file" in chosen_item else [])
                current_images, current_metas = _resolve_image_files(f_list, is_lunar=False)
                # Enrich metas with card details
                for m in current_metas:
                    m.update({
                        "name": chosen_item.get("name", m.get("filename", "")),
                        "sensor": chosen_item.get("sensor", m.get("sensor", "")),
                        "resolution": chosen_item.get("resolution", m.get("resolution_m", "")),
                        "date": chosen_item.get("date", m.get("acquisition_date", "")),
                        "location": chosen_item.get("location", m.get("crs", "")),
                        "provenance": chosen_item.get("provenance", m.get("source_dataset", "")),
                    })
            else:
                default_query = selected_preset.get("query", "")
                f_list = selected_preset.get("sample_files") or selected_preset.get("files") or []
                current_images, current_metas = _resolve_image_files(f_list, is_lunar=False)

            # Auto-sync text area with chosen image/preset query if preset or image changed
            current_choice_key = f"earth_{selected_idx}_{selected_img_idx}"
            if st.session_state.get("last_earth_choice_key") != current_choice_key:
                st.session_state["last_earth_choice_key"] = current_choice_key
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
                "Choose Chandrayaan-2 Target Category:",
                range(len(lunar_options)),
                format_func=lambda i: lunar_options[i],
                index=0,
                key="lunar_scenario_dropdown",
            )
            selected_lunar = lunar_presets[selected_lunar_idx]

            # Track scenario change to reset selected image index
            if st.session_state.get("last_lunar_preset_idx") != selected_lunar_idx:
                st.session_state["last_lunar_preset_idx"] = selected_lunar_idx
                st.session_state["selected_lunar_img_idx"] = 0

            # Visual Multiple Image Options Picker for Lunar Targets
            lunar_items = selected_lunar.get("sample_items") or []
            selected_lunar_img_idx = st.session_state.get("selected_lunar_img_idx", 0)

            if lunar_items:
                selected_lunar_img_idx = render_image_picker_gallery(
                    sample_items=lunar_items,
                    selected_idx=selected_lunar_img_idx,
                    key_prefix=f"lunar_scen_{selected_lunar_idx}",
                    resolve_thumb_fn=_resolve_single_thumb,
                )
                st.session_state["selected_lunar_img_idx"] = selected_lunar_img_idx

                chosen_lunar_item = lunar_items[selected_lunar_img_idx]
                default_query = chosen_lunar_item.get("query") or selected_lunar.get("query", "")
                l_files = [chosen_lunar_item["file"]] if "file" in chosen_lunar_item else []
                current_images, current_metas = _resolve_image_files(l_files, is_lunar=True)
                for m in current_metas:
                    m.update({
                        "name": chosen_lunar_item.get("name", m.get("filename", "")),
                        "sensor": chosen_lunar_item.get("sensor", "Chandrayaan-2 OHRC"),
                        "resolution": chosen_lunar_item.get("resolution", "0.25 m"),
                        "date": chosen_lunar_item.get("date", "Archival"),
                        "location": chosen_lunar_item.get("location", "Moon"),
                        "provenance": chosen_lunar_item.get("provenance", "ISRO / ISSDC PRADAN"),
                    })
            else:
                default_query = selected_lunar.get("query", "")
                l_files = [selected_lunar["file"]] if "file" in selected_lunar else []
                current_images, current_metas = _resolve_image_files(l_files, is_lunar=True)
                for m in current_metas:
                    m.update({
                        "name": selected_lunar.get("label", m.get("filename", "")),
                        "sensor": "Chandrayaan-2 OHRC",
                        "resolution": "0.25 m",
                        "date": "Archival Observation",
                        "location": "Moon (Lunar Surface)",
                        "provenance": "ISRO / ISSDC PRADAN Archive",
                    })

            # Auto-sync text area with chosen lunar image/preset query
            current_lunar_choice_key = f"lunar_{selected_lunar_idx}_{selected_lunar_img_idx}"
            if st.session_state.get("last_lunar_choice_key") != current_lunar_choice_key:
                st.session_state["last_lunar_choice_key"] = current_lunar_choice_key
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

# ---- Single Analysis Layout (60/40 Mission Workspace) ------------------
else:
    with st.container(border=True):
        col_workspace_img, col_workspace_report = st.columns([0.58, 0.42], gap="medium")

        with col_workspace_img:
            # Interactive Layer Controls if overlay is present
            has_overlay = (
                st.session_state.pipeline_result is not None
                and st.session_state.pipeline_result.get("overlay") is not None
            )
            layer_choice = "Base Satellite Raster"
            if has_overlay:
                layer_col1, layer_col2 = st.columns([0.55, 0.45])
                with layer_col1:
                    st.markdown(
                        '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.6px; padding-top:4px;">'
                        'Imagery & Detection Overlay'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                with layer_col2:
                    layer_choice = st.radio(
                        "Layer View:",
                        options=["Base Raster", "Grounding Mask"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key="viewer_layer_toggle_choice",
                    )

            if has_overlay and layer_choice == "Grounding Mask":
                st.image(
                    st.session_state.pipeline_result["overlay"],
                    caption="Visual Grounding & Morphological Activation Overlay",
                    use_container_width=True,
                )
                m_single = st.session_state.metas[0] if st.session_state.metas else {}
                st.markdown(
                    f'<div class="preview-hero-meta-bar">'
                    f'  <span class="preview-hero-chip"><strong>Active Layer:</strong> Model Detection Overlay</span>'
                    f'  <span class="preview-hero-chip"><strong>Sensor:</strong> {m_single.get("sensor", "Satellite")}</span>'
                    f'  <span class="preview-hero-chip"><strong>Status:</strong> {st.session_state.pipeline_result.get("confidence_tag", "Verified")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                render_image_preview(st.session_state.images, st.session_state.metas)

            st.write("")
            st.markdown(
                '<div style="font-size:0.85rem; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.6px; margin: 2px 0 6px 0;">'
                'Natural Language Query & Command Console'
                '</div>',
                unsafe_allow_html=True,
            )

            # Contextual Query Button Routing (Part 16)
            st.markdown(
                '<div class="query-chips-title">Contextual Quick Query Commands:</div>',
                unsafe_allow_html=True,
            )
            chip_col1, chip_col2, chip_col3 = st.columns(3)

            is_lunar_mode = (mode == "Lunar analysis")
            is_bitemporal = len(st.session_state.images) == 2 and any(
                "t1" in str(m.get("filename", "")).lower() or "levir" in str(m.get("source_dataset", "")).lower()
                for m in st.session_state.metas
            )
            is_multimodal = len(st.session_state.images) == 2 and not is_bitemporal
            is_sar = (
                len(st.session_state.images) == 1
                and any("sar" in str(m.get("sensor", "")).lower() or "sentinel-1" in str(m.get("sensor", "")).lower()
                for m in st.session_state.metas)
            )

            if is_lunar_mode:
                with chip_col1:
                    if st.button("+ Detect Craters", key="chip_lunar_craters", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Detect prominent impact crater candidates across the scene."
                        st.rerun()
                    if st.button("+ Analyze Crater Rims", key="chip_lunar_rims", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Analyze crater rim morphology, slope gradients, and rim degradation."
                        st.rerun()
                with chip_col2:
                    if st.button("+ Find Boulders", key="chip_lunar_boulders", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Locate boulder clusters and fragmented rock populations."
                        st.rerun()
                    if st.button("+ Analyze Shadows", key="chip_lunar_shadows", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify deep shadow zones and potential permanently shadowed regions (PSR)."
                        st.rerun()
                with chip_col3:
                    if st.button("+ Describe Terrain", key="chip_lunar_terrain", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Describe the lunar surface morphology and regolith micro-relief texture."
                        st.rerun()
                    if st.button("+ Detect Ejecta", key="chip_lunar_ejecta", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify high-albedo ejecta blanket patterns and radial impact rays."
                        st.rerun()

            elif is_multimodal:
                with chip_col1:
                    if st.button("+ Cross-Modal Analysis", key="chip_mm_cross", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Perform joint optical-SAR cross-modal analysis on this co-registered pair."
                        st.rerun()
                    if st.button("+ Compare Modalities", key="chip_mm_comp", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Compare optical reflectance features against SAR radar backscatter signatures."
                        st.rerun()
                with chip_col2:
                    if st.button("+ Detect Change", key="chip_mm_change", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Detect physical structural discrepancies between optical and SAR observations."
                        st.rerun()
                    if st.button("+ Identify Built-Up", key="chip_mm_built", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify built-up urban structures using SAR double-bounce reinforcement."
                        st.rerun()
                with chip_col3:
                    if st.button("+ Describe Scene", key="chip_mm_desc", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Describe the terrain and land cover through fused multimodal observation."
                        st.rerun()
                    if st.button("+ Locate Water", key="chip_mm_water", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify water bodies using specular radar nulls and NDWI spectral response."
                        st.rerun()

            elif is_bitemporal:
                with chip_col1:
                    if st.button("+ Detect Changes", key="chip_cd_change", use_container_width=True):
                        st.session_state["main_query_text_area"] = "What changed between these two dates? Has new construction occurred?"
                        st.rerun()
                    if st.button("+ New Buildings", key="chip_cd_bldgs", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Locate all newly built structures between Observation T1 and T2."
                        st.rerun()
                with chip_col2:
                    if st.button("+ Compare Pre/Post", key="chip_cd_comp", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Compare pre-event and post-event surface characteristics."
                        st.rerun()
                    if st.button("+ Infrastructure", key="chip_cd_infra", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Assess transportation network and building footprint expansion."
                        st.rerun()
                with chip_col3:
                    if st.button("+ Describe Scene", key="chip_cd_desc", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Describe the bi-temporal environmental and anthropogenic alterations."
                        st.rerun()
                    if st.button("+ Vegetation Shift", key="chip_cd_veg", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Assess vegetation clearing and seasonal biomass fluctuations."
                        st.rerun()

            elif is_sar:
                with chip_col1:
                    if st.button("+ Structural Features", key="chip_sar_struct", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Detect double-bounce structural features and high backscatter urban targets."
                        st.rerun()
                    if st.button("+ Analyze Flood Signal", key="chip_sar_flood", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Extract flood inundation areas using low-backscatter specular radar returns."
                        st.rerun()
                with chip_col2:
                    if st.button("+ Detect Change", key="chip_sar_change", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Detect structural deformation and radar cross-section anomalies."
                        st.rerun()
                    if st.button("+ Compare Backscatter", key="chip_sar_back", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Evaluate VV and VH polarimetric backscatter distributions in decibels."
                        st.rerun()
                with chip_col3:
                    if st.button("+ Describe Scene", key="chip_sar_desc", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Describe surface roughness and radar scattering mechanisms across the scene."
                        st.rerun()
                    if st.button("+ Water Boundaries", key="chip_sar_water", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Delineate dark calm water boundaries in this radar observation."
                        st.rerun()

            else:
                # Default Earth Optical
                with chip_col1:
                    if st.button("+ Locate Buildings", key="chip_opt_bldgs", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Locate all buildings and built-up structures in this scene."
                        st.rerun()
                    if st.button("+ Detect Changes", key="chip_opt_change", use_container_width=True):
                        st.session_state["main_query_text_area"] = "What changed between these observation dates? Has new construction occurred?"
                        st.rerun()
                with chip_col2:
                    if st.button("+ Find Water", key="chip_opt_water", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify flooded water bodies and coastal drainage networks."
                        st.rerun()
                    if st.button("+ Analyze Veg.", key="chip_opt_veg", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Assess vegetation density and compute physical spectral indices."
                        st.rerun()
                with chip_col3:
                    if st.button("+ Describe Scene", key="chip_opt_desc", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Describe the land use and dominant terrain features in this scene."
                        st.rerun()
                    if st.button("+ Identify Roads", key="chip_opt_roads", use_container_width=True):
                        st.session_state["main_query_text_area"] = "Identify major roads, highways, and transport corridors."
                        st.rerun()

            st.write("")
            active_query = st.text_area(
                "Query Prompt:",
                value=default_query,
                height=90,
                label_visibility="collapsed",
                key="main_query_text_area",
            )

            st.write("")
            run_analysis_btn = st.button(
                "RUN SATQUERY ANALYSIS →",
                type="primary",
                use_container_width=True,
                key="run_single_analysis_btn",
            )
            st.caption(
                "<span style='color:#64748b; font-size:0.8rem; font-family:ui-monospace, monospace;'>"
                "&bull; Autonomous Router &rarr; Specialist Pipeline &rarr; Deterministic Signal Verification"
                "</span>",
                unsafe_allow_html=True,
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
                        st.rerun()

        with col_workspace_report:
            if st.session_state.pipeline_result:
                render_scientific_report_panel(
                    st.session_state.pipeline_result,
                    st.session_state.metas,
                    is_lunar=(mode == "Lunar analysis"),
                )
            elif st.session_state.error_message:
                st.error(st.session_state.error_message)
            else:
                render_staging_telemetry_panel(
                    st.session_state.metas,
                    is_lunar=(mode == "Lunar analysis"),
                    query=active_query,
                )

