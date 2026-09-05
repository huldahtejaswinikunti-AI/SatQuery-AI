from satquery.router.task_router import TaskRouter
from satquery.router.task_types import TaskType
from satquery.validator.schemas import ValidationResult

def test_router_caption():
    val = ValidationResult(is_valid=True, num_images=1, detected_configuration="single_image")
    res = TaskRouter().route(val, "Describe the land cover in this image")
    assert res.task == TaskType.SINGLE_CAPTION

def test_router_grounding():
    val = ValidationResult(is_valid=True, num_images=1, detected_configuration="single_image")
    res = TaskRouter().route(val, "Highlight the water body")
    assert res.task == TaskType.GROUNDING
    assert "water body" in res.target_phrase.lower()

def test_router_change():
    val = ValidationResult(is_valid=True, num_images=2, detected_configuration="bitemporal_pair")
    res = TaskRouter().route(val, "What changed between dates?")
    assert res.task == TaskType.BITEMPORAL_CHANGE

def test_router_fusion():
    val = ValidationResult(is_valid=True, num_images=2, detected_configuration="optical_sar_pair")
    res = TaskRouter().route(val, "Analyze ground surface")
    assert res.task == TaskType.OPTICAL_SAR_FUSION
