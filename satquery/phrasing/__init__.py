"""Phrasing layer for SatQuery AI."""

__all__ = ["phrase", "PhrasingLLM"]


def __getattr__(name: str):
    """Lazy import to keep module import fast and model loading on-demand."""
    if name in ("phrase", "PhrasingLLM"):
        from satquery.phrasing.phrasing_llm import PhrasingLLM, phrase
        if name == "phrase":
            return phrase
        return PhrasingLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
