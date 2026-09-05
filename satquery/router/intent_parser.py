from satquery.router.task_types import RoutingDecision, TaskType
from satquery.validator.schemas import ValidationResult

class IntentParser:
    def parse(self, query: str, validation_result: ValidationResult) -> RoutingDecision:
        q = query.lower()
        if "change" in q:
            task = TaskType.BITEMPORAL_CHANGE
        elif "segment" in q or "where" in q or "highlight" in q:
            task = TaskType.GROUNDING
        elif "fuse" in q or "sar" in q:
            task = TaskType.OPTICAL_SAR_FUSION
        elif "describe" in q:
            task = TaskType.SINGLE_CAPTION
        else:
            task = TaskType.SINGLE_VQA
        return RoutingDecision(task=task, confidence=0.75, reasoning="Intent fallback parsed.", invoked_tools=["intent_parser", task.value])
