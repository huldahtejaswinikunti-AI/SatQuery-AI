from __future__ import annotations
from typing import Any

class PhrasingLLM:
    def phrase_response(self, task_type: str, query: str, structured_facts: dict[str, Any], confidence_tag: str, confidence_score: float) -> str:
        if task_type == "change_vqa":
            summary = structured_facts.get("change_summary", "Surface transitions detected.")
            frac = structured_facts.get("change_fraction", 0.0) * 100
            b_d = structured_facts.get("built_up_delta", 0.0) * 100
            v_d = structured_facts.get("vegetation_delta", 0.0) * 100
            lines = [
                "**Bi-Temporal Change Analysis:**",
                f"Approximately **{frac:.1f}%** of the monitored area underwent significant land-cover transition.",
                f"- {summary}"
            ]
            if abs(b_d) > 1.0: lines.append(f"- Net built-up footprint shifted by **{b_d:+.1f}%**.")
            if abs(v_d) > 1.0: lines.append(f"- Vegetation cover shifted by **{v_d:+.1f}%**.")
            return "\n".join(lines)

        elif task_type == "optical_sar_fusion":
            cloud = structured_facts.get("cloud_fraction", 0.0) * 100
            water = structured_facts.get("water_fraction", 0.0) * 100
            built = structured_facts.get("built_up_fraction", 0.0) * 100
            reasons = structured_facts.get("stated_reasons", [])
            lines = [
                "**Optical?SAR Cross-Modal Fusion Summary:**",
                f"- **Water Extent:** {water:.1f}% (cross-checked via specular reflection & NDWI).",
                f"- **Built-Up Fabric:** {built:.1f}% (verified via double-bounce backscatter & NDBI).",
                f"- **Cloud Cover:** {cloud:.1f}%."
            ]
            if reasons: lines.append(f"\n*Grounding Reason:* {reasons[0]}")
            return "\n".join(lines)

        elif task_type == "grounding":
            t = structured_facts.get("target_prompt", "target")
            expl = structured_facts.get("verification_explanation", "")
            return f"Successfully localized **'{t}'** across the imagery.\n\n*Verification:* {expl}" if expl else f"Localized '{t}'."

        elif task_type == "single_caption":
            return structured_facts.get("caption", "Remote sensing scene description.")

        vqa = structured_facts.get("vqa_answer", "")
        expl = structured_facts.get("verification_explanation", "")
        return f"{vqa}\n\n*Evidence Grounding:* {expl}" if expl else vqa
