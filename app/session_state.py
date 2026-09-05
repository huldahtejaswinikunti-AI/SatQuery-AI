import streamlit as st

def init_session_state() -> None:
    if "images" not in st.session_state: st.session_state.images = []
    if "metas" not in st.session_state: st.session_state.metas = []
    if "pipeline_result" not in st.session_state: st.session_state.pipeline_result = None
    if "query_input" not in st.session_state: st.session_state.query_input = ""

def reset_session_state() -> None:
    st.session_state.images = []
    st.session_state.metas = []
    st.session_state.pipeline_result = None
    st.session_state.query_input = ""
