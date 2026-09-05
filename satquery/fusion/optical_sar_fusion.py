from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from satquery.perception.cloud_mask import detect_cloud_mask
from satquery.perception.sar_backscatter import analyze_sar_backscatter
from satquery.perception.spectral_indices import compute_spectral_indices

@dataclass
class FusionResult:
    fused_water_mask: np.ndarray
    fused_built_up_mask: np.ndarray
    fused_vegetation_mask: np.ndarray
    water_fraction: float
    built_up_fraction: float
    vegetation_fraction: float
    cloud_fraction: float
    modality_weights: dict[str, float]
    agreement_score: float
    stated_reasons: list[str]
    confidence: float
    confidence_level: str

class OpticalSARFusionEngine:
    def fuse(self, optical_arr: np.ndarray, sar_arr: np.ndarray) -> FusionResult:
        h = min(optical_arr.shape[0], sar_arr.shape[0])
        w = min(optical_arr.shape[1], sar_arr.shape[1])

        spec = compute_spectral_indices(optical_arr)
        cloud = detect_cloud_mask(optical_arr)
        sar = analyze_sar_backscatter(sar_arr)

        def _resize(m):
            from PIL import Image
            if m.shape[:2] != (h, w):
                return np.array(Image.fromarray(m.astype(np.uint8)*255).resize((w, h))) > 127
            return m

        opt_w, opt_b, opt_v = _resize(spec.water_mask), _resize(spec.built_up_mask), _resize(spec.vegetation_mask)
        cloud_m = _resize(cloud.cloud_mask)
        sar_w, sar_b = _resize(sar.sar_water_mask), _resize(sar.sar_built_up_mask)

        c_frac = cloud.cloud_fraction
        w_opt = max(0.2, 1.0 - c_frac)
        w_sar = 1.0 - w_opt
        tot = w_opt + w_sar
        w_opt, w_sar = round(w_opt / tot, 2), round(w_sar / tot, 2)

        fused_w = np.zeros((h, w), dtype=bool)
        fused_w[~cloud_m] = opt_w[~cloud_m] | sar_w[~cloud_m]
        fused_w[cloud_m] = sar_w[cloud_m]

        fused_b = np.zeros((h, w), dtype=bool)
        fused_b[~cloud_m] = opt_b[~cloud_m] | sar_b[~cloud_m]
        fused_b[cloud_m] = sar_b[cloud_m]

        fused_v = opt_v.copy()
        fused_v[cloud_m] = False

        clear = ~cloud_m
        if np.any(clear):
            agree = (np.mean(opt_w[clear] == sar_w[clear]) + np.mean(opt_b[clear] == sar_b[clear])) / 2.0
            agree = round(float(agree), 3)
        else:
            agree = 0.50

        reasons = []
        if c_frac > 0.20:
            reasons.append(f"Cloud obscuration ({c_frac*100:.1f}%) detected in optical image. SAR radar penetration utilized to map surface features.")
            conf = 0.88
            lvl = "high_sar_penetration"
        elif agree >= 0.80:
            reasons.append(f"High cross-modal agreement ({agree*100:.1f}%) between optical reflectance and SAR backscatter.")
            conf = 0.95
            lvl = "high_cross_verified"
        else:
            reasons.append(f"Localized divergence between optical and SAR signals ({agree*100:.1f}% agreement). Assigning lower confidence.")
            conf = 0.75
            lvl = "lower_confidence_disagreement"

        total_p = float(h * w)
        return FusionResult(
            fused_water_mask=fused_w, fused_built_up_mask=fused_b, fused_vegetation_mask=fused_v,
            water_fraction=round(float(np.sum(fused_w))/total_p, 4),
            built_up_fraction=round(float(np.sum(fused_b))/total_p, 4),
            vegetation_fraction=round(float(np.sum(fused_v))/total_p, 4),
            cloud_fraction=round(c_frac, 4),
            modality_weights={"optical": w_opt, "sar": w_sar},
            agreement_score=agree, stated_reasons=reasons, confidence=conf, confidence_level=lvl,
        )
