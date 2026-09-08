"""Generate a comprehensive set of synthetic remote sensing demo samples for SatQuery AI.

Produces 24 realistic images across 16 scenarios covering:
- single_optical (5 samples)
- single_sar (3 samples)
- optical_sar_pairs (4 pairs = 8 images)
- bitemporal_pairs (4 pairs = 8 images)

Outputs both PNG (preview/standard) and georeferenced GeoTIFF (.tif) files,
plus a complete data/demo_samples/metadata.json catalog.
Requires zero internet connection (100% self-contained for offline SIH venue).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _save_geotiff(
    path: Path,
    array: np.ndarray,
    crs: str = "EPSG:32643",
    resolution: float = 10.0,
    west: float = 450000.0,
    north: float = 2100000.0,
) -> None:
    """Save array as GeoTIFF using rasterio if available, else PIL TIFF."""
    if HAS_RASTERIO:
        h, w = array.shape[:2]
        bands = array.shape[2] if array.ndim == 3 else 1
        transform = from_origin(west, north, resolution, resolution)
        dtype = "uint8" if array.dtype == np.uint8 else "float32"

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            else:
                for b in range(bands):
                    dst.write(array[:, :, b], b + 1)
    else:
        # Fallback to standard TIFF via Pillow
        im = Image.fromarray(array)
        im.save(path, format="TIFF")


def _add_speckle(arr: np.ndarray, std: float = 0.15) -> np.ndarray:
    """Simulate radar multiplicative speckle using gamma/Rayleigh model."""
    noise = np.random.gamma(shape=4.0, scale=0.25, size=arr.shape)
    speckled = arr.astype(np.float32) * noise
    return np.clip(speckled, 0, 255).astype(np.uint8)


def generate_samples(base_dir: str | Path = "data/demo_samples") -> None:
    base = Path(base_dir)
    np.random.seed(42)

    subdirs = ["single_optical", "single_sar", "optical_sar_pairs", "bitemporal_pairs"]
    for sub in subdirs:
        (base / sub).mkdir(parents=True, exist_ok=True)

    h, w = 256, 256
    y, x = np.mgrid[:h, :w]

    metadata_records: list[dict[str, Any]] = []

    def register_sample(
        subdir: str,
        filename_base: str,
        array: np.ndarray,
        task_type: str,
        sensor: str,
        acq_date: str,
        bands: list[str],
        crs: str = "EPSG:32643",
        res_m: float = 10.0,
    ) -> None:
        png_path = base / subdir / f"{filename_base}.png"
        tif_path = base / subdir / f"{filename_base}.tif"

        Image.fromarray(array).save(png_path)
        _save_geotiff(tif_path, array, crs=crs, resolution=res_m)

        metadata_records.append(
            {
                "file": f"{filename_base}.png",
                "tif_file": f"{filename_base}.tif",
                "rel_path": f"{subdir}/{filename_base}.png",
                "category": subdir,
                "task_type": task_type,
                "sensor": sensor,
                "acquisition_date": acq_date,
                "bands": bands,
                "channels": len(bands),
                "shape": [h, w, len(bands)] if array.ndim == 3 else [h, w, 1],
                "crs": crs,
                "resolution_m": res_m,
                "synthetic": True,
            }
        )

    # =========================================================================
    # 1. Single Optical (5 samples)
    # =========================================================================

    # 1.1 Coastal port
    port = np.full((h, w, 3), [28, 107, 160], dtype=np.uint8)  # Deep blue ocean
    land_mask = x >= 140
    port[land_mask] = [46, 125, 50]  # Green inland
    pier1 = (x >= 100) & (x < 140) & (y >= 80) & (y <= 100)
    pier2 = (x >= 90) & (x < 140) & (y >= 160) & (y <= 180)
    port[pier1 | pier2] = [180, 185, 190]  # Concrete jetties
    # Berths & cargo cranes (orange/yellow highlights)
    port[90:95, 120:135] = [230, 120, 20]
    port[170:175, 110:125] = [230, 120, 20]
    register_sample(
        "single_optical",
        "sample_coastal_port",
        port,
        "single_caption",
        "Sentinel-2 MSI",
        "2026-03-15",
        ["Red", "Green", "Blue"],
    )

    # 1.2 Urban grounding
    urban = np.full((h, w, 3), [195, 190, 180], dtype=np.uint8)  # Urban beige
    # Road network grid
    urban[::32, :] = [60, 60, 65]
    urban[:, ::32] = [60, 60, 65]
    # Industrial warehouse (bright white roof)
    wh_mask = (y >= 70) & (y <= 130) & (x >= 70) & (x <= 150)
    urban[wh_mask] = [245, 245, 250]
    # Parking lot (dark asphalt)
    pl_mask = (y >= 135) & (y <= 175) & (x >= 70) & (x <= 150)
    urban[pl_mask] = [75, 75, 80]
    # Tiny cars in parking lot
    urban[145:170:6, 75:145:8] = [220, 20, 20]
    register_sample(
        "single_optical",
        "sample_urban_grounding",
        urban,
        "grounding",
        "Cartosat-3 PAN",
        "2026-02-10",
        ["Red", "Green", "Blue"],
        res_m=0.28,
    )

    # 1.3 Agricultural fields
    agri = np.zeros((h, w, 3), dtype=np.uint8)
    # Patchwork grid of varying crop hues
    colors = [
        [34, 139, 34],  # Healthy crop green
        [154, 205, 50],  # Light pasture
        [189, 183, 107],  # Senescent / ripe grain
        [139, 90, 43],  # Bare fallow soil
    ]
    for r in range(4):
        for c in range(4):
            color = colors[(r * 3 + c) % len(colors)]
            agri[r * 64 : (r + 1) * 64, c * 64 : (c + 1) * 64] = color
    # Irrigation canal running across
    agri[124:132, :] = [30, 90, 150]
    register_sample(
        "single_optical",
        "sample_agri_fields",
        agri,
        "single_caption",
        "Sentinel-2 MSI",
        "2026-01-20",
        ["B4", "B3", "B2"],
    )

    # 1.4 Forest boundary
    forest = np.full((h, w, 3), [20, 80, 30], dtype=np.uint8)  # Dense dark canopy
    # Clear-cut boundary
    clear_cut = (y >= 120) & (x >= 60)
    forest[clear_cut] = [160, 140, 105]  # Brown clearing
    # Logging road
    forest[116:122, :] = [120, 100, 80]
    register_sample(
        "single_optical",
        "sample_forest_boundary",
        forest,
        "single_caption",
        "ResourceSat-2 LISS-4",
        "2025-11-18",
        ["NIR", "Red", "Green"],
        res_m=5.0,
    )

    # 1.5 Port complex (stretch query sample)
    complex_port = port.copy()
    # 6 docked vessels of varying sizes
    vessels = [
        (85, 95, 12, 4),
        (165, 80, 18, 5),
        (165, 115, 15, 4),
        (60, 50, 24, 7),
        (210, 70, 20, 6),
        (130, 40, 10, 3),
    ]
    for vy, vx, vh, vw in vessels:
        complex_port[vy : vy + vh, vx : vx + vw] = [235, 240, 245]
        # Subtle wake behind vessel
        if vx > 20:
            complex_port[vy : vy + vh, vx - 15 : vx] = [120, 180, 220]
    register_sample(
        "single_optical",
        "sample_port_complex",
        complex_port,
        "compound_reasoning",
        "Cartosat-3 PAN",
        "2026-04-02",
        ["Red", "Green", "Blue"],
        res_m=0.5,
    )

    # =========================================================================
    # 2. Single SAR (3 samples)
    # =========================================================================

    # 2.1 Estuary (specular dark water vs speckled mangrove banks)
    sar_estuary = np.full((h, w), 95, dtype=np.uint8)
    river = (y >= 100 + 20 * np.sin(x / 30.0)) & (y <= 160 + 20 * np.sin(x / 30.0))
    sar_estuary[river] = 12  # Specular reflection (dark)
    sar_estuary = _add_speckle(sar_estuary)
    register_sample(
        "single_sar",
        "sample_sar_estuary",
        sar_estuary,
        "sar_analysis",
        "Sentinel-1 C-SAR",
        "2026-02-14",
        ["VV"],
    )

    # 2.2 Urban backscatter grid (intense double-bounce cardinal points)
    sar_urban = np.full((h, w), 45, dtype=np.uint8)
    for ry in range(20, h - 20, 24):
        for rx in range(20, w - 20, 24):
            sar_urban[ry : ry + 8, rx : rx + 8] = 245  # Strong dihedral/trihedral reflection
    sar_urban = _add_speckle(sar_urban, std=0.2)
    register_sample(
        "single_sar",
        "sample_sar_urban",
        sar_urban,
        "sar_analysis",
        "EOS-04 C-SAR",
        "2026-01-11",
        ["VV", "VH"],
    )

    # 2.3 Flood inundation radar
    sar_flood = np.full((h, w), 85, dtype=np.uint8)
    flood_basin = (x >= 40) & (x <= 210) & (y >= 60) & (y <= 200)
    sar_flood[flood_basin] = 15  # Water surface -> very low backscatter
    # Village mounds elevated above water
    sar_flood[90:110, 110:130] = 210
    sar_flood[150:170, 80:100] = 210
    sar_flood = _add_speckle(sar_flood)
    register_sample(
        "single_sar",
        "sample_sar_flood",
        sar_flood,
        "sar_analysis",
        "Sentinel-1 C-SAR",
        "2025-09-03",
        ["VV"],
    )

    # =========================================================================
    # 3. Optical + SAR Pairs (4 pairs = 8 images)
    # =========================================================================

    # 3.1 Monsoon cloudy optical + penetrating SAR
    cloudy_opt = np.full((h, w, 3), [40, 130, 50], dtype=np.uint8)  # Rural vegetation
    # Heavy cloud sheet across top 60%
    cloud_mask = (y <= 160) | ((x >= 80) & (y <= 200))
    cloudy_opt[cloud_mask] = np.clip(
        cloudy_opt[cloud_mask].astype(int) + 175 + np.random.randint(-15, 15, cloudy_opt[cloud_mask].shape),
        0,
        255,
    ).astype(np.uint8)
    # Dark cloud shadow
    shadow_mask = (y >= 165) & (y <= 185) & (x >= 90) & (x <= 180)
    cloudy_opt[shadow_mask] = [15, 30, 20]

    sar_penetrating = np.full((h, w), 80, dtype=np.uint8)
    # Flooded plain hidden under clouds
    sar_penetrating[(y >= 70) & (y <= 150) & (x >= 50) & (x <= 170)] = 14  # Flat water
    # Railway embankment and concrete structure visible in SAR
    sar_penetrating[110:114, :] = 225
    sar_penetrating = _add_speckle(sar_penetrating)

    register_sample(
        "optical_sar_pairs",
        "sample_monsoon_cloudy",
        cloudy_opt,
        "optical_sar_fusion",
        "Sentinel-2 MSI",
        "2025-08-22",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "optical_sar_pairs",
        "sample_sar_vv_vh",
        sar_penetrating,
        "optical_sar_fusion",
        "Sentinel-1 C-SAR",
        "2025-08-22",
        ["VV", "VH"],
    )

    # Also save backwards-compatible aliases
    Image.fromarray(cloudy_opt).save(base / "optical_sar_pairs" / "optical_cloudy_s2.png")
    Image.fromarray(sar_penetrating).save(base / "optical_sar_pairs" / "sar_penetrating_s1.png")

    # 3.2 City optical + City SAR
    city_opt = np.full((h, w, 3), [170, 165, 160], dtype=np.uint8)
    city_opt[::24, :] = [50, 50, 55]
    city_opt[:, ::24] = [50, 50, 55]
    city_sar = sar_urban.copy()

    register_sample(
        "optical_sar_pairs",
        "sample_city_opt",
        city_opt,
        "optical_sar_fusion",
        "Cartosat-2S",
        "2026-01-15",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "optical_sar_pairs",
        "sample_city_sar",
        city_sar,
        "optical_sar_fusion",
        "EOS-04 C-SAR",
        "2026-01-15",
        ["VV"],
    )

    # 3.3 Coastal optical + Coastal SAR
    coast_opt = port.copy()
    coast_sar = np.full((h, w), 25, dtype=np.uint8)
    coast_sar[land_mask] = 110
    coast_sar[pier1 | pier2] = 230
    coast_sar = _add_speckle(coast_sar)

    register_sample(
        "optical_sar_pairs",
        "sample_coast_opt",
        coast_opt,
        "optical_sar_fusion",
        "Sentinel-2 MSI",
        "2026-03-01",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "optical_sar_pairs",
        "sample_coast_sar",
        coast_sar,
        "optical_sar_fusion",
        "Sentinel-1 C-SAR",
        "2026-03-01",
        ["VV", "VH"],
    )

    # 3.4 Mountain haze optical + Mountain SAR
    mountain_opt = np.full((h, w, 3), [100, 115, 90], dtype=np.uint8)
    mountain_opt = np.clip(mountain_opt + (y * 0.4).astype(int)[..., np.newaxis], 0, 255).astype(np.uint8)  # Valley gradient
    # Atmospheric haze
    mountain_opt[:, :] = np.clip(mountain_opt.astype(int) + 60, 0, 255).astype(np.uint8)

    mountain_sar = np.full((h, w), 90, dtype=np.uint8)
    mountain_sar[y < 120] = 210  # Foreshortened mountain slopes
    mountain_sar[y >= 180] = 25  # Radar shadow behind ridge
    mountain_sar = _add_speckle(mountain_sar)

    register_sample(
        "optical_sar_pairs",
        "sample_haze_opt",
        mountain_opt,
        "optical_sar_fusion",
        "ResourceSat-2",
        "2025-12-05",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "optical_sar_pairs",
        "sample_mountain_sar",
        mountain_sar,
        "optical_sar_fusion",
        "Sentinel-1 C-SAR",
        "2025-12-05",
        ["VV"],
    )

    # =========================================================================
    # 4. Bi-Temporal Pairs (4 pairs = 8 images)
    # =========================================================================

    # 4.1 LEVIR-CD pre/post
    levir_pre = np.full((h, w, 3), [135, 175, 105], dtype=np.uint8)  # Grassland
    levir_pre[::64, :] = [110, 105, 95]  # Dirt path
    levir_post = levir_pre.copy()
    # 4 new rectangular buildings + asphalt driveway
    bldgs = [(40, 50, 35, 45), (40, 150, 35, 50), (140, 50, 40, 45), (140, 150, 45, 55)]
    for by, bx, bh, bw in bldgs:
        levir_post[by : by + bh, bx : bx + bw] = [225, 220, 215]  # White roofs
        # Shadow
        levir_post[by + bh : by + bh + 6, bx : bx + bw] = [50, 60, 40]
    levir_post[85:135, 100:115] = [65, 65, 70]  # Paved driveway

    register_sample(
        "bitemporal_pairs",
        "sample_levir_pre",
        levir_pre,
        "bitemporal_change",
        "Cartosat-2",
        "2023-01-10",
        ["Red", "Green", "Blue"],
        res_m=0.8,
    )
    register_sample(
        "bitemporal_pairs",
        "sample_levir_post",
        levir_post,
        "bitemporal_change",
        "Cartosat-2",
        "2025-01-12",
        ["Red", "Green", "Blue"],
        res_m=0.8,
    )

    # Also save backwards-compatible aliases
    Image.fromarray(levir_pre).save(base / "bitemporal_pairs" / "levir_sample_t1_before.png")
    Image.fromarray(levir_post).save(base / "bitemporal_pairs" / "levir_sample_t2_after.png")

    # 4.2 Urban expansion T1/T2
    urban_t1 = agri.copy()
    urban_t2 = agri.copy()
    # Conversion of top-right quadrant from farmland to dense built-up
    urban_t2[:128, 128:] = [190, 185, 180]
    urban_t2[:128:20, 128:] = [55, 55, 60]
    urban_t2[:128, 128::20] = [55, 55, 60]

    register_sample(
        "bitemporal_pairs",
        "sample_urban_t1",
        urban_t1,
        "bitemporal_change",
        "Sentinel-2 MSI",
        "2022-04-18",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "bitemporal_pairs",
        "sample_urban_t2",
        urban_t2,
        "bitemporal_change",
        "Sentinel-2 MSI",
        "2026-04-20",
        ["Red", "Green", "Blue"],
    )

    # 4.3 Flood pre/post
    flood_pre = np.full((h, w, 3), [60, 140, 70], dtype=np.uint8)  # Dry floodplain
    flood_pre[120:135, :] = [35, 95, 160]  # Normal narrow river channel

    flood_post = flood_pre.copy()
    # Extensive dark floodwaters expanding across 50% of the scene
    flood_post[60:190, :] = [25, 65, 110]
    flood_post[90:160, 100:150] = [70, 130, 60]  # Unsubmerged island

    register_sample(
        "bitemporal_pairs",
        "sample_flood_pre",
        flood_pre,
        "bitemporal_change",
        "ResourceSat-2 AWiFS",
        "2025-05-10",
        ["NIR", "Red", "Green"],
        res_m=56.0,
    )
    register_sample(
        "bitemporal_pairs",
        "sample_flood_post",
        flood_post,
        "bitemporal_change",
        "ResourceSat-2 AWiFS",
        "2025-08-15",
        ["NIR", "Red", "Green"],
        res_m=56.0,
    )

    # 4.4 Seasonal vegetation change (summer vs winter)
    veg_summer = np.full((h, w, 3), [30, 150, 45], dtype=np.uint8)  # Lush photosynthetic green
    veg_summer[x % 40 < 4, :] = [20, 90, 30]  # Treeline hedgerows

    veg_winter = np.full((h, w, 3), [175, 160, 110], dtype=np.uint8)  # Dry dormant / fallow yellow-brown
    veg_winter[x % 40 < 4, :] = [110, 95, 65]

    register_sample(
        "bitemporal_pairs",
        "sample_veg_summer",
        veg_summer,
        "bitemporal_change",
        "Sentinel-2 MSI",
        "2025-07-28",
        ["Red", "Green", "Blue"],
    )
    register_sample(
        "bitemporal_pairs",
        "sample_veg_winter",
        veg_winter,
        "bitemporal_change",
        "Sentinel-2 MSI",
        "2026-01-14",
        ["Red", "Green", "Blue"],
    )

    # =========================================================================
    # Write metadata.json catalog
    # =========================================================================
    catalog = {
        "dataset": "SatQuery AI Curated Remote Sensing Demo Samples",
        "problem_statement": "SIH 2026 PS 26167 (ISRO/SAC)",
        "version": "2.0.0",
        "total_samples": len(metadata_records),
        "synthetic": True,
        "samples": metadata_records,
    }

    meta_file = base / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Generated {len(metadata_records)} samples in '{base}'. Metadata saved to '{meta_file}'.")


if __name__ == "__main__":
    generate_samples()
