"""Session-state helpers for the SatQuery AI Streamlit app.

Preserves previous results across reruns and supports smooth switching
between demo presets and free-text input during a live presentation.
"""

from __future__ import annotations

import streamlit as st


_DEFAULTS = {
    "images": [],
    "metas": [],
    "pipeline_result": None,
    "previous_result": None,
    "query_input": "",
    "query_source": "freetext",   # "freetext" | "preset"
    "demo_preset_index": 0,
    "analysis_mode": "Earth analysis",  # "Earth analysis" | "Lunar analysis"
    "batch_queue": [],
    "batch_results": [],
    "is_running": False,
    "error_message": None,
}


def init_session_state() -> None:
    """Initialise all session-state keys with defaults (idempotent)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_session_state() -> None:
    """Clear everything back to defaults."""
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default


def store_result(result: dict) -> None:
    """Save a pipeline result, keeping the previous one for comparison."""
    if st.session_state.pipeline_result is not None:
        st.session_state.previous_result = st.session_state.pipeline_result
    st.session_state.pipeline_result = result
    st.session_state.is_running = False
    st.session_state.error_message = None


def store_error(message: str) -> None:
    """Record an error message without losing the previous result."""
    st.session_state.error_message = message
    st.session_state.is_running = False


def set_query_from_preset(query: str, index: int) -> None:
    """Update the query from a demo preset dropdown."""
    st.session_state.query_input = query
    st.session_state.query_source = "preset"
    st.session_state.demo_preset_index = index


def set_query_freetext(query: str) -> None:
    """Update the query from free-text input."""
    st.session_state.query_input = query
    st.session_state.query_source = "freetext"
