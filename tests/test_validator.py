import numpy as np
from satquery.validator.input_validator import InputValidator
from satquery.validator.schemas import Modality

def test_validator_empty():
    res = InputValidator().validate([], [])
    assert not res.is_valid
    assert "No images" in res.error_message

def test_validator_single():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = InputValidator().validate([img], [{"filename": "opt.png", "format": "PNG"}])
    assert res.is_valid
    assert res.num_images == 1
    assert res.detected_configuration == "single_image"

def test_validator_fusion():
    img_opt = np.zeros((64, 64, 4), dtype=np.float32)
    img_sar = np.zeros((64, 64, 2), dtype=np.float32)
    res = InputValidator().validate([img_opt, img_sar], [{"filename": "s2.tif"}, {"filename": "s1.tif"}])
    assert res.is_valid
    assert res.detected_configuration == "optical_sar_pair"

def test_validator_change():
    t1 = np.zeros((64, 64, 3), dtype=np.uint8)
    t2 = np.zeros((64, 64, 3), dtype=np.uint8)
    res = InputValidator().validate([t1, t2], [{"filename": "t1.png"}, {"filename": "t2.png"}])
    assert res.is_valid
    assert res.detected_configuration == "bitemporal_pair"
