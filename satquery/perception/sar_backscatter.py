from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from satquery.utils.config import settings
from satquery.utils.image_utils import extract_sar_bands

@dataclass
class SARBackscatterResult:
    vv_db: np.ndarray
    vh_db: np.ndarray
    sar_water_mask: np.ndarray
    sar_built_up_mask: np.ndarray
    sar_water_fraction: float
    sar_built_up_fraction: float
    mean_vv_db: float
    mean_vh_db: float

def linear_to_db(linear: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    val = np.maximum(linear.astype(np.float32), eps)
    return 10.0 * np.log10(val)

def analyze_sar_backscatter(sar_arr: np.ndarray, is_already_db: bool = False) -> SARBackscatterResult:
    bands = extract_sar_bands(sar_arr)
    vv, vh = bands["vv"], bands["vh"]

    if not is_already_db:
        if np.min(vv) >= 0.0:
            vv_norm = vv / (np.max(vv) + 1e-6) if np.max(vv) > 1.0 else vv
            vh_norm = vh / (np.max(vh) + 1e-6) if np.max(vh) > 1.0 else vh
            vv_db = -30.0 + vv_norm * 30.0
            vh_db = -35.0 + vh_norm * 30.0
        else:
            vv_db, vh_db = vv, vh
    else:
        vv_db, vh_db = vv, vh

    thresh = settings.sar
    sar_water_mask = vv_db <= thresh.vv_water_max_db
    sar_built_up_mask = (vv_db >= thresh.vv_urban_min_db) | (vh_db >= thresh.vh_urban_min_db)

    tot = float(vv_db.shape[0] * vv_db.shape[1])
    return SARBackscatterResult(
        vv_db=vv_db, vh_db=vh_db,
        sar_water_mask=sar_water_mask, sar_built_up_mask=sar_built_up_mask,
        sar_water_fraction=round(float(np.sum(sar_water_mask)) / tot, 4),
        sar_built_up_fraction=round(float(np.sum(sar_built_up_mask)) / tot, 4),
        mean_vv_db=round(float(np.mean(vv_db)), 2),
        mean_vh_db=round(float(np.mean(vh_db)), 2),
    )
