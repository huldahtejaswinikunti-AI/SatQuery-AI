from __future__ import annotations
from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(page_title="SatQuery AI", page_icon="???", layout="wide", initial_sidebar_state="expanded")

from app.session_state import init_session_state, reset_session_state
from app.ui_components import render_header, render_confidence_badge, render_image_preview, render_execution_trace, render_download_buttons
from satquery.pipeline.executor import PipelineExecutor
from satquery.utils.geo_io import load_image_as_array

css = Path(__file__).parent / "assets" / "styles.css"
if css.exists():
    with open(css) as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()

@st.cache_resource
def get_executor(): return PipelineExecutor()
executor = get_executor()

render_header()

with st.sidebar:
    st.header("1. Input Configuration")
    demo_option = st.selectbox(
        "Load Curated Demo Scenario:",
        [
            "Custom Upload",
            "Demo 1: Single Optical VQA (Sentinel-2)",
            "Demo 2: Open-Vocabulary Grounding (Water Body)",
            "Demo 3: Bi-temporal Change Detection (LEVIR-CD)",
            "Demo 4: Cloud-Penetrating Optical-SAR Fusion (S1+S2)",
            "Demo 5: Cross-Verification Disagreement Probe"
        ]
    )
    if st.button("Reset / Clear", use_container_width=True):
        reset_session_state()
        st.rerun()

    uploaded = st.file_uploader("Upload Imagery (1 or 2 files)", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=True)

def generate_demo(opt: str):
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]

    if "Demo 1" in opt:
        # Synthetic Sentinel-2-like 4-band image:
        # channels 0-2 = visible bands, channel 3 = NIR
        img = np.full((h, w, 4), 0.2, dtype=np.float32)

        # Upper region = vegetation with high NIR
        img[:130, :, 3] = 0.8

        # Lower region = water
        img[130:, :, 1] = 0.7

        return (
            [img],
            [{"filename": "demo_s2.tif", "format": "GeoTIFF"}],
            "What is the primary land-cover type and is surface water present?"
        )

    elif "Demo 2" in opt:
        img = np.full((h, w, 3), 120, dtype=np.uint8)

        circle = (x - 128) ** 2 + (y - 128) ** 2 < 50 ** 2
        img[circle] = [20, 80, 200]

        return (
            [img],
            [{"filename": "demo_grounding.png", "format": "PNG"}],
            "Highlight the water body referred to in the scene."
        )

    elif "Demo 3" in opt:
        t1 = np.full((h, w, 3), [40, 150, 40], dtype=np.uint8)

        t2 = t1.copy()
        t2[100:200, 100:200] = [210, 205, 200]

        return (
            [t1, t2],
            [
                {"filename": "t1_before.png", "format": "PNG"},
                {"filename": "t2_after.png", "format": "PNG"}
            ],
            "What changed between these two dates and has built-up area increased?"
        )

    elif "Demo 4" in opt:
        # Optical image with 4 bands
        opt_arr = np.full((h, w, 4), 0.25, dtype=np.float32)

        # Upper region = cloud
        opt_arr[:130, :, :] = 0.92

        # SAR image with 2 channels
        sar_arr = np.full((h, w, 2), -20.0, dtype=np.float32)

        # Urban structures beneath cloud
        urban_mask = (y < 100) & (x < 120)
        sar_arr[urban_mask] = -4.0

        return (
            [opt_arr, sar_arr],
            [
                {"filename": "opt_cloudy.tif", "format": "TIFF"},
                {"filename": "sar_grd.tif", "format": "TIFF"}
            ],
            "Use optical and SAR together to identify built-up structures hidden under cloud."
        )

    else:
        img = np.full((h, w, 3), 180, dtype=np.uint8)

        return (
            [img],
            [{"filename": "probe.png", "format": "PNG"}],
            "Is this entire area covered in deep open water?"
        )

if demo_option != "Custom Upload":
    imgs, metas, default_q = generate_demo(demo_option)
    st.session_state.images = imgs
    st.session_state.metas = metas
else:
    default_q = "Describe the land-cover and features visible in this satellite imagery."

if uploaded:
    imgs, metas = [], []
    for f in uploaded:
        arr, m = load_image_as_array(f)
        imgs.append(arr); metas.append(m)
    st.session_state.images = imgs
    st.session_state.metas = metas

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    render_image_preview(st.session_state.images, st.session_state.metas)
    q = st.text_area("Natural Language Query:", value=default_q, height=85)
    run = st.button("?? Analyze & Ground", type="primary", use_container_width=True)

with col_r:
    st.subheader("Analysis & Evidence Grounding")
    if run:
        if not st.session_state.images:
            st.error("Please provide at least one satellite image.")
        else:
            with st.spinner("Executing pipeline with RS cross-verification..."):
                st.session_state.pipeline_result = executor.run(st.session_state.images, st.session_state.metas, q)

    if st.session_state.pipeline_result:
        res = st.session_state.pipeline_result
        render_confidence_badge(res.get("confidence_tag", "UNKNOWN"), res.get("confidence_score", 0.0))
        st.markdown(res.get("answer", ""))
        if res.get("overlay") is not None:
            st.image(res["overlay"], caption="Visual Grounding & Analysis Overlay", use_container_width=True)
        render_execution_trace(res.get("trace", {}))
        render_download_buttons(res)
    else:
        st.info("Select a demo scenario or upload your own imagery, then click **'Analyze & Ground'**.")
