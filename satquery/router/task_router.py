from __future__ import annotations
import re
from satquery.router.task_types import RoutingDecision, TaskType
from satquery.validator.schemas import ValidationResult

class TaskRouter:
    GROUNDING_KEYWORDS = ["highlight", "segment", "locate", "where is", "show me", "mask", "find the", "bound"]
    CAPTION_KEYWORDS = ["describe", "caption", "summary", "overview", "what is in", "tell me about"]
    CHANGE_KEYWORDS = ["change", "difference", "before and after", "what changed", "increased", "decreased"]
    FUSION_KEYWORDS = ["optical and sar", "sar and optical", "fuse", "fusion", "cross-modal", "radar and optical"]

    def route(self, validation_result: ValidationResult, query: str) -> RoutingDecision:
        q_lower = query.strip().lower()
        config = validation_result.detected_configuration

        if config == "optical_sar_pair":
            return RoutingDecision(task=TaskType.OPTICAL_SAR_FUSION, confidence=0.98, reasoning="Paired optical and SAR input.", invoked_tools=["spectral_indices", "sar_backscatter", "optical_sar_fusion"])
        if config == "bitemporal_pair":
            return RoutingDecision(task=TaskType.BITEMPORAL_CHANGE, confidence=0.98, reasoning="Bi-temporal pair detected.", invoked_tools=["tinycd_change", "cross_verification"])
        if any(k in q_lower for k in self.FUSION_KEYWORDS):
            return RoutingDecision(task=TaskType.OPTICAL_SAR_FUSION, confidence=0.92, reasoning="Fusion query requested.", invoked_tools=["spectral_indices", "sar_backscatter", "optical_sar_fusion"])

        for kw in self.GROUNDING_KEYWORDS:
            if kw in q_lower:
                clean = re.sub(r"[^\w\s]", "", query).strip()
                idx = clean.lower().find(kw)
                target = clean[idx + len(kw):].strip() if idx != -1 else "target"
                target = re.sub(r"^(the|a|an|in|of)\s+", "", target, flags=re.IGNORECASE)
                return RoutingDecision(task=TaskType.GROUNDING, confidence=0.95, reasoning="Grounding request.", target_phrase=target or "target", invoked_tools=["clipseg_grounding", "spectral_indices"])

        if any(kw in q_lower for kw in self.CAPTION_KEYWORDS) or len(q_lower) == 0:
            return RoutingDecision(task=TaskType.SINGLE_CAPTION, confidence=0.90, reasoning="Captioning request.", invoked_tools=["geochat_vqa"])

        return RoutingDecision(task=TaskType.SINGLE_VQA, confidence=0.88, reasoning="VQA query.", invoked_tools=["geochat_vqa", "spectral_indices", "cross_verification"])
