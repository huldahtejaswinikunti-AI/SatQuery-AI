import numpy as np
from satquery.cross_verification.verifier import CrossVerifier
from satquery.perception.spectral_indices import SpectralIndicesResult

def test_verifier_agreement():
    v = CrossVerifier()
    spec = SpectralIndicesResult(
        ndvi=np.zeros((4, 4)), ndwi=np.ones((4, 4)), ndbi=np.zeros((4, 4)),
        vegetation_mask=np.zeros((4, 4), bool), water_mask=np.ones((4, 4), bool), built_up_mask=np.zeros((4, 4), bool),
        vegetation_fraction=0.0, water_fraction=1.0, built_up_fraction=0.0
    )
    res = v.verify_vqa_claim("Yes, water is present.", 0.9, spec, "Is there water?")
    assert res.is_cross_verified
    assert res.confidence_tag == "high_cross_verified"

def test_verifier_disagreement():
    v = CrossVerifier()
    spec = SpectralIndicesResult(
        ndvi=np.zeros((4, 4)), ndwi=np.full((4, 4), -0.5), ndbi=np.zeros((4, 4)),
        vegetation_mask=np.zeros((4, 4), bool), water_mask=np.zeros((4, 4), bool), built_up_mask=np.zeros((4, 4), bool),
        vegetation_fraction=0.0, water_fraction=0.0, built_up_fraction=0.0
    )
    res = v.verify_vqa_claim("Yes, a large lake is detected.", 0.9, spec, "Is there water?")
    assert not res.is_cross_verified
    assert res.confidence_tag == "lower_confidence_disagreement"
