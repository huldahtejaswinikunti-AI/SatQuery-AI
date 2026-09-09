"""Markdown report generator - detailed, professional analysis reports.

Converts an execution trace, phrased answer, verified facts, and optional
overlay references into a comprehensive downloadable Markdown report with
spectral analysis tables, classification breakdowns, and verification details.

Public API
----------
generate_report(trace, phrased_answer, overlay_refs, verified_facts) -> str
"""

from __future__ import annotations

from typing import Any


def generate_report(
    trace: dict[str, Any],
    phrased_answer: str,
    overlay_refs: list[str] | None = None,
    verified_facts: dict[str, Any] | None = None,
) -> str:
    """Generate a detailed Markdown report from pipeline outputs."""
    vf = verified_facts or {}
    lines: list[str] = []

    lines.append("# SatQuery AI - Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {trace.get('timestamp', 'N/A')}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(phrased_answer)
    lines.append("")

    lines.append("## Task Information")
    lines.append("")
    lines.append(f"- **Task Type:** `{trace.get('task', 'N/A')}`")
    lines.append(f"- **Router Confidence:** {trace.get('confidence', 'N/A')}")
    params = trace.get("parameters", {})
    if params.get("query"):
        lines.append(f"- **Query:** {params['query']}")
    if params.get("image_count"):
        lines.append(f"- **Images Analyzed:** {params['image_count']}")
    if params.get("input_type"):
        lines.append(f"- **Input Type:** {params['input_type']}")
    lines.append("")

    spectral = vf.get("spectral_summary", {})
    if spectral:
        lines.append("## Spectral Analysis")
        lines.append("")
        lines.append("Physical spectral indices computed from calibrated sensor data:")
        lines.append("")
        lines.append("| Index | Value | Interpretation |")
        lines.append("|-------|-------|----------------|")

        ndvi = spectral.get("ndvi_mean", 0)
        ndwi = spectral.get("ndwi_mean", 0)
        ndbi = spectral.get("ndbi_mean", 0)

        if ndvi > 0.4:
            ndvi_interp = "Dense, healthy vegetation"
        elif ndvi > 0.2:
            ndvi_interp = "Moderate vegetation coverage"
        elif ndvi > 0.1:
            ndvi_interp = "Sparse vegetation"
        else:
            ndvi_interp = "Bare soil / non-vegetated"
        lines.append(f"| **NDVI** (Vegetation) | `{ndvi:+.4f}` | {ndvi_interp} |")

        if ndwi > 0.2:
            ndwi_interp = "Open water body"
        elif ndwi > 0.0:
            ndwi_interp = "Wet surface / partial water"
        else:
            ndwi_interp = "Dry land surface"
        lines.append(f"| **NDWI** (Water) | `{ndwi:+.4f}` | {ndwi_interp} |")

        if ndbi > 0.1:
            ndbi_interp = "Built-up / urban area"
        elif ndbi > 0.0:
            ndbi_interp = "Mixed built-up and natural"
        else:
            ndbi_interp = "Natural land cover"
        lines.append(f"| **NDBI** (Built-up) | `{ndbi:+.4f}` | {ndbi_interp} |")
        lines.append("")

        veg_f = spectral.get("vegetation_fraction", 0)
        wat_f = spectral.get("water_fraction", 0)
        blt_f = spectral.get("built_up_fraction", 0)
        other_f = max(0, 1.0 - veg_f - wat_f - blt_f)

        lines.append("### Scene Composition")
        lines.append("")
        lines.append("| Land Type | Coverage |")
        lines.append("|-----------|----------|")
        lines.append(f"| Vegetation | {veg_f*100:.1f}% |")
        lines.append(f"| Water Bodies | {wat_f*100:.1f}% |")
        lines.append(f"| Built-up Areas | {blt_f*100:.1f}% |")
        lines.append(f"| Other / Unclassified | {other_f*100:.1f}% |")
        lines.append("")

    top_k = vf.get("top_k", [])
    if top_k:
        lines.append("## Land Cover Classification")
        lines.append("")
        lines.append("Multi-label classification via fine-tuned ResNet-18 on BigEarthNet-S2:")
        lines.append("")
        lines.append("| Rank | Land Cover Class | Probability |")
        lines.append("|------|------------------|-------------|")
        for i, entry in enumerate(top_k, 1):
            prob_pct = entry.get("probability", 0) * 100
            lines.append(f"| {i} | {entry.get('class_name', 'Unknown')} | {prob_pct:.1f}% |")
        lines.append("")

    conf_tag = vf.get("confidence_tag", "")
    reason = vf.get("reason", "")
    if conf_tag:
        lines.append("## Cross-Verification")
        lines.append("")
        tag_display = conf_tag.replace("_", " ").title()
        lines.append(f"- **Status:** {tag_display}")
        agreed_str = "Agreed" if vf.get("agreed", False) else "Disagreement"
        lines.append(f"- **Agreement:** {agreed_str}")
        if reason:
            lines.append(f"- **Explanation:** {reason}")
        raw_conf = vf.get("raw_confidence")
        if raw_conf is not None:
            lines.append(f"- **Specialist Confidence:** {float(raw_conf)*100:.1f}%")
        lines.append("")

    change = vf.get("change_summary", {})
    if change:
        lines.append("## Change Detection Summary")
        lines.append("")
        pct = change.get("pct_area_changed", 0)
        lines.append(f"- **Area Changed:** {pct:.2f}%")
        cb = change.get("class_before", [])
        ca = change.get("class_after", [])
        if cb:
            lines.append(f"- **Before Classes:** {', '.join(cb)}")
        if ca:
            lines.append(f"- **After Classes:** {', '.join(ca)}")
        if cb and ca:
            appeared = set(ca) - set(cb)
            disappeared = set(cb) - set(ca)
            if appeared:
                lines.append(f"- **New Classes:** {', '.join(appeared)}")
            if disappeared:
                lines.append(f"- **Lost Classes:** {', '.join(disappeared)}")
        lines.append("")

    tools = trace.get("tools_invoked", [])
    if tools:
        lines.append("## Methodology")
        lines.append("")
        lines.append("Pipeline execution sequence:")
        lines.append("")
        lines.append("| Step | Tool | Status |")
        lines.append("|------|------|--------|")
        for idx, tool in enumerate(tools, start=1):
            lines.append(f"| {idx} | `{tool}` | Completed |")
        lines.append("")

    if params:
        lines.append("## Analysis Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, value in params.items():
            lines.append(f"| `{key}` | {value} |")
        lines.append("")

    if overlay_refs:
        lines.append("## Visualisations")
        lines.append("")
        for ref in overlay_refs:
            lines.append(f"![Overlay]({ref})")
            lines.append("")

    lines.append("---")
    lines.append(
        "*Report generated by SatQuery AI. "
        "All spectral measurements derived from calibrated sensor data. "
        "See the execution trace JSON for full audit details.*"
    )
    lines.append("")

    return "\n".join(lines)
