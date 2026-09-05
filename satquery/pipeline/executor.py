from __future__ import annotations
import time, uuid
from typing import Any
import numpy as np

from satquery.validator.input_validator import InputValidator
from satquery.router.task_router import TaskRouter
from satquery.router.task_types import TaskType
from satquery.perception.spectral_indices import compute_spectral_indices
from satquery.specialists.geochat_vqa import GeoChatSpecialist
from satquery.specialists.clipseg_grounding import CLIPSegGroundingSpecialist
from satquery.specialists.tinycd_change import TinyCDSpecialist
from satquery.fusion.optical_sar_fusion import OpticalSARFusionEngine
from satquery.cross_verification.verifier import CrossVerifier
from satquery.phrasing.phrasing_llm import PhrasingLLM
from satquery.pipeline.execution_trace import ExecutionTrace, ToolStep
from satquery.utils.image_utils import to_display_rgb
from satquery.utils.overlay import create_mask_overlay, create_change_overlay, create_side_by_side

class PipelineExecutor:
    def __init__(self):
        self.validator = InputValidator()
        self.router = TaskRouter()
        self.geochat = GeoChatSpecialist()
        self.clipseg = CLIPSegGroundingSpecialist()
        self.tinycd = TinyCDSpecialist()
        self.fusion = OpticalSARFusionEngine()
        self.verifier = CrossVerifier()
        self.phrasing = PhrasingLLM()

    def run(self, images: list[np.ndarray], metas: list[dict[str, Any]], query: str) -> dict[str, Any]:
        start = time.perf_counter()
        trace_id = str(uuid.uuid4())[:8]
        steps = []

        # 1. Validation
        val = self.validator.validate(images, metas, query)
        steps.append(ToolStep(tool_name="InputValidator", output_summary={"is_valid": val.is_valid}))
        if not val.is_valid:
            tr = ExecutionTrace(trace_id=trace_id, task="error", query=query, tools_invoked=["InputValidator"], confidence=0.0, confidence_tag="error", steps=steps)
            return {"answer": f"Input validation failed: {val.error_message}", "confidence_score": 0.0, "confidence_tag": "error", "overlay": None, "trace": tr.to_dict()}

        # 2. Routing
        decision = self.router.route(val, query)
        steps.append(ToolStep(tool_name="TaskRouter", output_summary={"task": decision.task.value}))
        task = decision.task
        primary = images[0]

        facts = {}
        overlay = None
        conf_score = 0.90
        conf_tag = "high_rule_based"

        if task == TaskType.SINGLE_VQA:
            spec = compute_spectral_indices(primary)
            steps.append(ToolStep(tool_name="spectral_indices", output_summary={"veg": spec.vegetation_fraction}))
            vqa_res = self.geochat.answer_query(primary, query)
            steps.append(ToolStep(tool_name="GeoChatSpecialist"))
            verif = self.verifier.verify_vqa_claim(vqa_res["answer"], vqa_res["confidence"], spec, query)
            steps.append(ToolStep(tool_name="CrossVerifier", output_summary={"cross_verified": verif.is_cross_verified}))
            conf_score = verif.confidence_score
            conf_tag = verif.confidence_tag
            facts = {"vqa_answer": vqa_res["answer"], "verification_explanation": verif.explanation}
            overlay = to_display_rgb(primary)

        elif task == TaskType.SINGLE_CAPTION:
            spec = compute_spectral_indices(primary)
            steps.append(ToolStep(tool_name="spectral_indices"))
            cap_res = self.geochat.generate_caption(primary)
            steps.append(ToolStep(tool_name="GeoChatSpecialist"))
            conf_score = cap_res["confidence"]
            conf_tag = "high_cross_verified"
            facts = {"caption": cap_res["caption"]}
            overlay = to_display_rgb(primary)

        elif task == TaskType.GROUNDING:
            target = decision.target_phrase or "target region"
            seg = self.clipseg.segment(primary, prompt=target)
            steps.append(ToolStep(tool_name="CLIPSegSpecialist"))
            spec = compute_spectral_indices(primary)
            verif = self.verifier.verify_grounding_mask(seg["binary_mask"], target, spec)
            steps.append(ToolStep(tool_name="CrossVerifier"))
            conf_score = verif.confidence_score
            conf_tag = verif.confidence_tag
            facts = {"target_prompt": target, "verification_explanation": verif.explanation}
            base = to_display_rgb(primary)
            color = (0, 191, 255) if "water" in target else (46, 204, 113) if "veg" in target else (255, 69, 0)
            overlay = create_mask_overlay(base, seg["probability_mask"], color=color)

        elif task == TaskType.BITEMPORAL_CHANGE:
            cd = self.tinycd.detect_change(images[0], images[1])
            steps.append(ToolStep(tool_name="TinyCDSpecialist"))
            conf_score = cd["confidence"]
            conf_tag = "high_rule_based"
            facts = cd
            rgb1 = to_display_rgb(images[0])
            rgb2 = to_display_rgb(images[1])
            over = create_change_overlay(rgb1, rgb2, cd["change_mask"])
            overlay = create_side_by_side([rgb1, rgb2, over], titles=["Before", "After", "Changes"])

        elif task == TaskType.OPTICAL_SAR_FUSION:
            opt = images[0] if val.images_metadata[0].modality != "sar" else images[1]
            sar = images[1] if val.images_metadata[0].modality != "sar" else images[0]
            fres = self.fusion.fuse(opt, sar)
            steps.append(ToolStep(tool_name="OpticalSARFusionEngine"))
            conf_score = fres.confidence
            conf_tag = fres.confidence_level
            facts = {
                "water_fraction": fres.water_fraction, "built_up_fraction": fres.built_up_fraction,
                "cloud_fraction": fres.cloud_fraction, "stated_reasons": fres.stated_reasons,
                "modality_weights": fres.modality_weights
            }
            opt_rgb = to_display_rgb(opt)
            sar_rgb = to_display_rgb(sar)
            fused_disp = opt_rgb.copy()
            fused_disp[fres.fused_water_mask] = [0, 150, 255]
            fused_disp[fres.fused_built_up_mask] = [255, 60, 60]
            overlay = create_side_by_side([opt_rgb, sar_rgb, fused_disp], titles=["Optical (S2)", "SAR (S1)", "Fused Analysis"])

        ans = self.phrasing.phrase_response(task.value, query, facts, conf_tag, conf_score)
        steps.append(ToolStep(tool_name="PhrasingLLM"))

        tr = ExecutionTrace(
            trace_id=trace_id, task=task.value, query=query,
            tools_invoked=[s.tool_name for s in steps],
            parameters={"config": val.detected_configuration, "num_images": len(images)},
            confidence=round(conf_score, 3), confidence_tag=conf_tag, steps=steps,
            metadata={"duration_ms": round((time.perf_counter() - start)*1000, 2)}
        )
        return {
            "answer": ans, "confidence_score": round(conf_score, 3), "confidence_tag": conf_tag,
            "overlay": overlay, "trace": tr.to_dict(), "structured_facts": facts, "query": query
        }
