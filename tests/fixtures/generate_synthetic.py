"""
Synthetic fixture generator for SatQuery AI.
Generates realistic multi-spectral Sentinel-2 bands and Sentinel-1 SAR arrays
with known spatial ground truths (water, vegetation, urban, cloud).
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image

def generate_fixtures(output_dir: Path | None = None) -> dict[str, Path]:
    if output_dir is None:
        base_path = Path(__file__).resolve().parent.parent.parent
        optical_dir = base_path / "data" / "demo_samples" / "single_optical"
        sar_dir = base_path / "data" / "demo_samples" / "single_sar"
    else:
        optical_dir = output_dir / "single_optical"
        sar_dir = output_dir / "single_sar"

    optical_dir.mkdir(parents=True, exist_ok=True)
    sar_dir.mkdir(parents=True, exist_ok=True)

    H, W = 128, 128
    np.random.seed(42)

    # Base noise
    noise = lambda scale: np.random.normal(0, scale, (H, W)).astype(np.float32)

    # 1. Optical Sentinel-2 bands (Surface reflectance 0.0 - 1.0)
    red = np.full((H, W), 0.15, dtype=np.float32) + noise(0.01)
    green = np.full((H, W), 0.15, dtype=np.float32) + noise(0.01)
    blue = np.full((H, W), 0.12, dtype=np.float32) + noise(0.01)
    nir = np.full((H, W), 0.20, dtype=np.float32) + noise(0.01)
    swir = np.full((H, W), 0.18, dtype=np.float32) + noise(0.01)

    # Zone 1: Water Body (Top-Left quadrant [0:64, 0:64])
    # Low NIR (absorbs strongly), higher Green than NIR -> positive NDWI, negative NDVI
    red[0:64, 0:64] = 0.04 + noise(0.005)[0:64, 0:64]
    green[0:64, 0:64] = 0.10 + noise(0.005)[0:64, 0:64]
    blue[0:64, 0:64] = 0.14 + noise(0.005)[0:64, 0:64]
    nir[0:64, 0:64] = 0.02 + noise(0.003)[0:64, 0:64]
    swir[0:64, 0:64] = 0.01 + noise(0.002)[0:64, 0:64]

    # Zone 2: Dense Forest / Vegetation (Bottom-Left quadrant [64:128, 0:64])
    # Very high NIR, low Red -> high NDVI (~0.7 - 0.8)
    red[64:128, 0:64] = 0.05 + noise(0.005)[64:128, 0:64]
    green[64:128, 0:64] = 0.18 + noise(0.008)[64:128, 0:64]
    blue[64:128, 0:64] = 0.04 + noise(0.004)[64:128, 0:64]
    nir[64:128, 0:64] = 0.65 + noise(0.015)[64:128, 0:64]
    swir[64:128, 0:64] = 0.12 + noise(0.008)[64:128, 0:64]

    # Zone 3: Built-up / Urban Fabric (Bottom-Right quadrant [64:128, 64:128])
    # High SWIR, moderate NIR -> positive NDBI
    red[64:128, 64:128] = 0.28 + noise(0.01)[64:128, 64:128]
    green[64:128, 64:128] = 0.25 + noise(0.01)[64:128, 64:128]
    blue[64:128, 64:128] = 0.24 + noise(0.01)[64:128, 64:128]
    nir[64:128, 64:128] = 0.26 + noise(0.01)[64:128, 64:128]
    swir[64:128, 64:128] = 0.45 + noise(0.015)[64:128, 64:128]

    # Zone 4: Cloud patch in Top-Right quadrant [10:45, 75:115]
    # High brightness in all optical bands, low saturation
    cloud_slice = (slice(10, 45), slice(75, 115))
    red[cloud_slice] = 0.85 + noise(0.02)[cloud_slice]
    green[cloud_slice] = 0.86 + noise(0.02)[cloud_slice]
    blue[cloud_slice] = 0.88 + noise(0.02)[cloud_slice]
    nir[cloud_slice] = 0.82 + noise(0.02)[cloud_slice]
    swir[cloud_slice] = 0.70 + noise(0.02)[cloud_slice]

    # Clip to valid reflectance range [0, 1]
    red = np.clip(red, 0.0, 1.0)
    green = np.clip(green, 0.0, 1.0)
    blue = np.clip(blue, 0.0, 1.0)
    nir = np.clip(nir, 0.0, 1.0)
    swir = np.clip(swir, 0.0, 1.0)

    # Save optical npz
    optical_npz_path = optical_dir / "sample_sentinel2_bands.npz"
    np.savez_compressed(
        optical_npz_path,
        red=red,
        green=green,
        blue=blue,
        nir=nir,
        swir=swir,
    )

    # Save RGB PNG preview
    rgb_img = np.stack([red, green, blue], axis=-1)
    rgb_uint8 = (np.clip(rgb_img * 255.0, 0, 255)).astype(np.uint8)
    png_path = optical_dir / "sample_coastal_port.png"
    Image.fromarray(rgb_uint8).save(png_path)

    # 2. Sentinel-1 SAR GRD arrays in dB
    # Typical values:
    # Water: VV <= -18 dB, VH <= -25 dB
    # Urban: VV >= -6 dB, VH >= -12 dB (double bounce)
    # Vegetation: VV ~ -12 dB, VH ~ -18 dB (diffuse volume)
    vv = np.full((H, W), -12.0, dtype=np.float32) + noise(0.5)
    vh = np.full((H, W), -18.0, dtype=np.float32) + noise(0.5)

    # Water zone
    vv[0:64, 0:64] = -22.0 + noise(0.8)[0:64, 0:64]
    vh[0:64, 0:64] = -28.0 + noise(0.8)[0:64, 0:64]

    # Forest zone
    vv[64:128, 0:64] = -11.0 + noise(0.5)[64:128, 0:64]
    vh[64:128, 0:64] = -16.0 + noise(0.5)[64:128, 0:64]

    # Built-up zone (double-bounce)
    vv[64:128, 64:128] = -5.0 + noise(0.8)[64:128, 64:128]
    vh[64:128, 64:128] = -11.0 + noise(0.8)[64:128, 64:128]

    # Cloud zone does NOT affect radar! Radar penetrates clouds.
    # Top-right quadrant [0:64, 64:128] is rural/mixed
    vv[0:64, 64:128] = -13.0 + noise(0.6)[0:64, 64:128]
    vh[0:64, 64:128] = -19.0 + noise(0.6)[0:64, 64:128]

    sar_npz_path = sar_dir / "sample_sentinel1_sar.npz"
    np.savez_compressed(
        sar_npz_path,
        vv=vv,
        vh=vh,
    )

    print(f"Generated synthetic optical: {optical_npz_path}")
    print(f"Generated synthetic optical PNG: {png_path}")
    print(f"Generated synthetic SAR: {sar_npz_path}")

    return {
        "optical_npz": optical_npz_path,
        "optical_png": png_path,
        "sar_npz": sar_npz_path,
    }

if __name__ == "__main__":
    generate_fixtures()
