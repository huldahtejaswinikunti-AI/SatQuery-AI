import numpy as np
from satquery.perception.sar_backscatter import analyze_sar_backscatter, linear_to_db

def test_linear_to_db():
    res = linear_to_db(np.array([1.0, 10.0]))
    assert np.isclose(res[0], 0.0, atol=1e-3)
    assert np.isclose(res[1], 10.0, atol=1e-3)

def test_sar_analysis():
    h, w = 16, 16
    vv = np.full((h, w), -20.0, dtype=np.float32)
    vh = np.full((h, w), -25.0, dtype=np.float32)
    sar = np.stack([vv, vh], axis=-1)
    res = analyze_sar_backscatter(sar, is_already_db=True)
    assert res.sar_water_fraction == 1.0
