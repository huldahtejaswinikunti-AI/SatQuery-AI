import io
import numpy as np
from PIL import Image
from satquery.utils.geo_io import load_image_as_array

def test_geo_io():
    img = Image.new("RGB", (16, 16), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    arr, meta = load_image_as_array(buf)
    assert arr.shape == (16, 16, 3)
    assert meta["channels"] == 3
