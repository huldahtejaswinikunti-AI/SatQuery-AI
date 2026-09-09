"""Unified spectral index interpretation and scene composition engine.

Provides deterministic, physical mapping from optical/infrared indices (NDVI,
NDWI, NDBI) to standardized descriptive categories and multi-bucket scene
composition narratives. This ensures complete consistency across pipeline
executors, visual question answering, captioning, and generated reports.
"""

from __future__ import annotations

from typing import Any, Tuple


# ---------------------------------------------------------------------------
# Individual Spectral Index Interpretation (Single Source of Truth)
# ---------------------------------------------------------------------------


def interpret_ndvi(ndvi: float) -> str:
    """Classify Normalized Difference Vegetation Index into physical land categories.

    Thresholds:
      > 0.4 : Dense, healthy vegetation
      > 0.2 : Moderate vegetation coverage
      > 0.1 : Sparse vegetation
      <= 0.1: Bare soil / non-vegetated
    """
    if ndvi > 0.4:
        return "Dense, healthy vegetation"
    elif ndvi > 0.2:
        return "Moderate vegetation coverage"
    elif ndvi > 0.1:
        return "Sparse vegetation"
    else:
        return "Bare soil / non-vegetated"


def interpret_ndwi(ndwi: float) -> str:
    """Classify Normalized Difference Water Index into aquatic surface categories.

    Thresholds:
      > 0.2 : Open water body
      > 0.0 : Wet surface / partial water
      <= 0.0: Dry land surface
    """
    if ndwi > 0.2:
        return "Open water body"
    elif ndwi > 0.0:
        return "Wet surface / partial water"
    else:
        return "Dry land surface"


def interpret_ndbi(ndbi: float | None) -> str:
    """Classify Normalized Difference Built-up Index into structural surface categories.

    Thresholds:
      > 0.1 : Built-up / urban area
      > 0.0 : Mixed built-up and natural
      <= 0.0: Natural land cover
    """
    if ndbi is None:
        return "Unavailable (requires SWIR band)"
    if ndbi > 0.1:
        return "Built-up / urban area"
    elif ndbi > 0.0:
        return "Mixed built-up and natural"
    else:
        return "Natural land cover"


# ---------------------------------------------------------------------------
# Scene Composition & Dominance Analysis (4-Bucket Majority Logic)
# ---------------------------------------------------------------------------


def compute_four_way_composition(
    veg_pct: float,
    water_pct: float,
    built_pct: float | None,
) -> dict[str, float | None]:
    """Compute 4-way terrain partition (vegetation / water / built-up / other)."""
    if built_pct is None:
        other_pct = max(0.0, round(100.0 - veg_pct - water_pct, 1))
        return {
            "vegetation": round(veg_pct, 1),
            "water": round(water_pct, 1),
            "built_up": None,
            "other": other_pct,
        }
    other_pct = max(0.0, round(100.0 - veg_pct - water_pct - built_pct, 1))
    return {
        "vegetation": round(veg_pct, 1),
        "water": round(water_pct, 1),
        "built_up": round(built_pct, 1),
        "other": other_pct,
    }


def describe_dominant_land_cover(
    veg_pct: float,
    water_pct: float,
    built_pct: float | None,
    ndvi: float,
    ndwi: float,
    ndbi: float | None,
) -> str:
    """Generate a mathematically sound natural-language description of scene dominance.

    Guarantees:
    - Never uses 'predominantly' unless an identified category achieves a strict > 50% majority.
    - If unclassified/other or mixed terrain holds the majority, uses hedged phrasing:
      'the most prominent identified land-cover signal is X (Y% of the scene)'.
    - Uses the exact physical interpretation labels from `interpret_ndvi`, `interpret_ndwi`, and `interpret_ndbi`.
    """
    comp = compute_four_way_composition(veg_pct, water_pct, built_pct)
    veg = comp["vegetation"]
    wat = comp["water"]
    blt = comp["built_up"]
    oth = comp["other"]

    ndvi_label = interpret_ndvi(ndvi)
    ndwi_label = interpret_ndwi(ndwi)
    ndbi_label = interpret_ndbi(ndbi)

    # 1. Strict Majority Check (> 50%)
    if veg > 50.0:
        return (
            f"The scene is predominantly vegetated ({veg}% coverage, NDVI: {ndvi:+.3f}), "
            f"indicating {ndvi_label.lower()}."
        )
    if wat > 50.0:
        return (
            f"The scene is predominantly water ({wat}% coverage, NDWI: {ndwi:+.3f}), "
            f"indicating {ndwi_label.lower()}."
        )
    if blt is not None and blt > 50.0 and ndbi is not None:
        return (
            f"The scene is predominantly built-up ({blt}% coverage, NDBI: {ndbi:+.3f}), "
            f"indicating {ndbi_label.lower()}."
        )

    # 2. Plurality / Identified Signal Check
    tracked = [
        ("vegetation", veg, "NDVI", ndvi, ndvi_label, 15.0),
        ("water", wat, "NDWI", ndwi, ndwi_label, 5.0),
    ]
    if blt is not None and ndbi is not None:
        tracked.append(("built-up", blt, "NDBI", ndbi, ndbi_label, 5.0))

    # Sort by fraction descending
    tracked.sort(key=lambda x: x[1], reverse=True)
    top_name, top_pct, top_idx_name, top_val, top_interp, min_thresh = tracked[0]

    if top_pct >= min_thresh:
        return (
            f"The most prominent identified land-cover signal is {top_name} "
            f"({top_pct}% coverage, {top_idx_name}: {top_val:+.3f}), indicating {top_interp.lower()}."
        )

    # 3. Mixed / Unclassified Baseline
    built_str = f", {blt}% built-up" if blt is not None else ""
    return (
        f"Mixed land cover: {veg}% vegetation, {wat}% water{built_str}, "
        f"{oth}% other/unclassified."
    )
