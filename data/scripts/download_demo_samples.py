from pathlib import Path
import json
import numpy as np
from PIL import Image

def generate_samples(base_dir="data/demo_samples"):
    base = Path(base_dir)
    for sub in ["single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    h, w = 256, 256
    y, x = np.ogrid[:h, :w]

    opt = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)
    opt[(y >= 100) & (y <= 160)] = [20, 80, 180]
    Image.fromarray(opt).save(base / "single_optical" / "sample_coastal_port.png")

    sar = np.full((h, w), 35, dtype=np.uint8)
    sar[y >= 120] = 8
    Image.fromarray(sar).save(base / "single_sar" / "sample_sar_estuary.png")

    opt_c = opt.copy()
    opt_c[y < 130] = 240
    sar_p = sar.copy()
    sar_p[(y < 100) & (x < 120)] = 230
    Image.fromarray(opt_c).save(base / "optical_sar_pairs" / "optical_cloudy_s2.png")
    Image.fromarray(sar_p).save(base / "optical_sar_pairs" / "sar_penetrating_s1.png")

    t1 = opt.copy()
    t2 = t1.copy()
    t2[100:200, 100:200] = [200, 200, 195]
    Image.fromarray(t1).save(base / "bitemporal_pairs" / "levir_sample_t1_before.png")
    Image.fromarray(t2).save(base / "bitemporal_pairs" / "levir_sample_t2_after.png")

    meta = {
        "dataset": "SatQuery Demo Samples",
        "samples": [
            {"id": "s1", "type": "single_optical", "file": "sample_coastal_port.png"},
            {"id": "s2", "type": "single_sar", "file": "sample_sar_estuary.png"},
            {"id": "s3", "type": "optical_sar_pair", "files": ["optical_cloudy_s2.png", "sar_penetrating_s1.png"]},
            {"id": "s4", "type": "bitemporal_pair", "files": ["levir_sample_t1_before.png", "levir_sample_t2_after.png"]}
        ]
    }
    with open(base / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Demo samples & metadata generated.")

if __name__ == "__main__":
    generate_samples()
