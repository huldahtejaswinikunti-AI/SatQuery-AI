"""Regression tests for Deterministic Spectral Dominance and Interpretation Consistency (Bug #3).

Verifies that:
1. When input composition has veg=23.7, water=7.1, built=7.1, other=62.1, ndvi=+0.0771:
   - The generated description does NOT contain "predominantly vegetated" (since 23.7% is not a majority).
   - The NDVI-derived phrase in the executive summary / caption matches the NDVI interpretation in the spectral table.
2. The interpretation function is shared and deterministic across executor and report generator.
"""

import pytest

from satquery.perception.spectral_interpretation import (
    interpret_ndvi,
    interpret_ndwi,
    interpret_ndbi,
    describe_dominant_land_cover,
    compute_four_way_composition,
)
from satquery.pipeline.report_generator import generate_report


def test_mediterranean_coastal_sample_no_predominantly_vegetated():
    """Exact numbers from the sample report: veg=23.7, water=7.1, built=7.1, other=62.1, ndvi=+0.0771.

    Must NOT claim 'predominantly vegetated' when other/unclassified is 62.1%.
    """
    veg_pct = 23.7
    water_pct = 7.1
    built_pct = 7.1
    ndvi = 0.0771
    ndwi = -0.1250
    ndbi = 0.0450

    # 1. Test four-way composition
    comp = compute_four_way_composition(veg_pct, water_pct, built_pct)
    assert comp["other"] == 62.1
    assert comp["vegetation"] == 23.7

    # 2. Test dominant land cover description
    desc = describe_dominant_land_cover(
        veg_pct=veg_pct,
        water_pct=water_pct,
        built_pct=built_pct,
        ndvi=ndvi,
        ndwi=ndwi,
        ndbi=ndbi,
    )

    # Must NOT claim predominantly vegetated
    assert "predominantly vegetated" not in desc.lower()
    assert "predominantly" not in desc.lower()

    # Must accurately describe the identified signal without overstating dominance
    assert "most prominent identified land-cover signal is vegetation" in desc or "vegetation (23.7%" in desc

    # 3. Test that NDVI interpretation is identical to what report_generator produces
    expected_ndvi_label = interpret_ndvi(ndvi)
    assert expected_ndvi_label == "Bare soil / non-vegetated"
    assert expected_ndvi_label.lower() in desc.lower()

    # 4. Generate report with these verified facts and verify mutual agreement
    verified_facts = {
        "answer": desc,
        "spectral_summary": {
            "ndvi_mean": ndvi,
            "ndwi_mean": ndwi,
            "ndbi_mean": ndbi,
            "vegetation_fraction": 0.237,
            "water_fraction": 0.071,
            "built_up_fraction": 0.071,
        },
    }
    trace = {
        "task": "single_image_caption",
        "tools_invoked": ["land_cover_specialist"],
        "parameters": {"query": "Summarize land cover."},
        "confidence": "0.95",
        "timestamp": "2026-09-10T00:00:00Z",
    }
    report = generate_report(trace, phrased_answer=desc, verified_facts=verified_facts)

    # Assert executive summary in report does not say predominantly vegetated
    assert "predominantly vegetated" not in report.lower()

    # Assert spectral table contains the exact same interpretation string
    assert f"| **NDVI** (Vegetation) | `{ndvi:+.4f}` | {expected_ndvi_label} |" in report


def test_majority_threshold_enforced():
    """Verify that 'predominantly' is only used when fraction strictly exceeds 50%."""
    # Case A: 48% vegetation, 52% others -> NOT predominantly
    desc_48 = describe_dominant_land_cover(
        veg_pct=48.0, water_pct=10.0, built_pct=10.0, ndvi=0.55, ndwi=-0.2, ndbi=-0.1,
    )
    assert "predominantly" not in desc_48.lower()

    # Case B: 55% vegetation -> Predominantly vegetated
    desc_55 = describe_dominant_land_cover(
        veg_pct=55.0, water_pct=10.0, built_pct=10.0, ndvi=0.55, ndwi=-0.2, ndbi=-0.1,
    )
    assert "predominantly vegetated" in desc_55.lower()
    assert "dense, healthy vegetation" in desc_55.lower()


def test_ndvi_threshold_continuity():
    """Verify thresholds produce correct standard classifications."""
    assert interpret_ndvi(0.65) == "Dense, healthy vegetation"
    assert interpret_ndvi(0.25) == "Moderate vegetation coverage"
    assert interpret_ndvi(0.15) == "Sparse vegetation"
    assert interpret_ndvi(0.05) == "Bare soil / non-vegetated"
    assert interpret_ndvi(-0.2) == "Bare soil / non-vegetated"
