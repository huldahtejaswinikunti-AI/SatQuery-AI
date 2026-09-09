"""Manual-Assist & Conversion Script for Chandrayaan-2 OHRC/TMC-2 Lunar Data.

This script guides researchers through downloading genuine Chandrayaan-2 lunar products
from the login-gated ISRO / ISSDC PRADAN portal and provides an automated PDS-4 (.IMG)
to standard GeoTIFF/PNG conversion pipeline.

Mandatory Acknowledgment (ISRO / ISSDC):
    "The research is based partially / to a significant extent on the results obtained
    from the Chandrayaan-II, second lunar mission of ISRO, archived at the Indian Space
    Science Data Centre (ISSDC)."

Usage:
    # Print download instructions:
    python data/scripts/download_chandrayaan_samples.py --instructions

    # Convert a downloaded PDS-4 .IMG file to PNG/GeoTIFF:
    python data/scripts/download_chandrayaan_samples.py --convert path/to/ch2_product.IMG --output data/demo_samples/lunar/sample.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def print_issdc_download_guide() -> None:
    """Display step-by-step instructions for acquiring Chandrayaan-2 data from ISSDC."""
    guide = """
================================================================================
CHANDRAYAAN-2 DATA ACQUISITION GUIDE (ISSDC PRADAN)
================================================================================

Step 1: Account Registration
  - Visit: https://pradan.issdc.gov.in
  - Click 'Register' and create a free researcher account.
  - Verify your email address and log in.

Step 2: Browse Orbital Footprints
  - For visual map browsing: https://chmapbrowse.issdc.gov.in
  - Select 'Chandrayaan-2' mission.
  - Choose Payload:
      * OHRC (Orbiter High Resolution Camera) — 0.25m/pixel (highest resolution)
      * TMC-2 (Terrain Mapping Camera 2) — 5.0m/pixel (stereo 3D coverage)
  - Zoom into target regions of interest (e.g. Boguslawsky, Shackleton Crater,
    Mare Serenitatis, or Von Kármán).

Step 3: Download Calibrated Products
  - Select your desired product granule (Level-1 or Level-2 calibrated).
  - Download the package containing:
      * Product Data file: `.IMG` (raw 16-bit or 32-bit raster)
      * PDS-4 Metadata label: `.xml` or `.lbl`

Step 4: Convert to SatQuery AI Format
  - Run this conversion script on your downloaded .IMG:
      python data/scripts/download_chandrayaan_samples.py --convert <path_to_IMG> --output data/demo_samples/lunar/<filename>.png

Step 5: Mandatory Citation Notice
  - Include the following acknowledgment in any publication or presentation:
    "The research is based partially / to a significant extent on the results obtained
    from the Chandrayaan-II, second lunar mission of ISRO, archived at the Indian
    Space Science Data Centre (ISSDC)."
================================================================================
"""
    print(guide)


def convert_pds4_img_to_raster(
    input_path: str | Path,
    output_path: str | Path,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Path:
    """Convert a Chandrayaan-2 PDS-4 .IMG raster to GeoTIFF or PNG.

    Parameters
    ----------
    input_path : str | Path
        Path to the Chandrayaan-2 .IMG file.
    output_path : str | Path
        Destination .png or .tif path.
    percentile_low : float
        Lower percentile for reflectance stretch (default: 1.0%).
    percentile_high : float
        Upper percentile for reflectance stretch (default: 99.0%).

    Returns
    -------
    Path
        Path to the converted raster file.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Chandrayaan-2 product file not found: {in_p}")

    # Method 1: Try pvl / planetaryimage / pdr if installed
    arr: np.ndarray | None = None
    try:
        import pvl  # type: ignore
        lbl_path = in_p.with_suffix(".xml") if in_p.with_suffix(".xml").exists() else in_p.with_suffix(".lbl")
        if lbl_path.exists():
            label = pvl.load(str(lbl_path))
            print(f"[PDS-4 Reader] Loaded PDS label from {lbl_path.name}")
    except ImportError:
        pass

    try:
        import rasterio  # type: ignore
        with rasterio.open(in_p) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
    except Exception:
        pass

    # Method 2: Standard PIL or binary fallback
    if arr is None:
        try:
            pil_img = Image.open(in_p)
            arr = np.array(pil_img).astype(np.float32)
        except Exception:
            raw = in_p.read_bytes()
            # Estimate square dimensions for 16-bit unsigned integers
            n_pix = len(raw) // 2
            dim = int(np.sqrt(n_pix))
            arr = np.frombuffer(raw[:dim*dim*2], dtype=np.uint16).reshape((dim, dim)).astype(np.float32)

    # Apply 2-98% or specified percentile contrast stretch
    valid = arr[~np.isnan(arr)]
    if len(valid) > 0:
        p_lo = np.percentile(valid, percentile_low)
        p_hi = np.percentile(valid, percentile_high)
        norm = np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6) * 255.0, 0, 255)
        uint8_data = np.nan_to_num(norm, nan=0).astype(np.uint8)
    else:
        uint8_data = np.zeros_like(arr, dtype=np.uint8)

    # Save to destination format
    if out_p.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            from rasterio.transform import from_origin
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                out_p, "w",
                driver="GTiff",
                height=uint8_data.shape[0],
                width=uint8_data.shape[1],
                count=1,
                dtype=uint8_data.dtype,
                transform=transform,
            ) as dst:
                dst.write(uint8_data, 1)
        except Exception:
            Image.fromarray(uint8_data).save(out_p)
    else:
        Image.fromarray(uint8_data).save(out_p)

    print(f"[Chandrayaan-2 Converter] Successfully exported {out_p.name} ({uint8_data.shape[1]}x{uint8_data.shape[0]})")
    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chandrayaan-2 ISSDC PRADAN Manual-Assist & Conversion Tool")
    parser.add_argument("--instructions", action="store_true", help="Print ISSDC download guide")
    parser.add_argument("--convert", type=str, help="Input .IMG file path to convert")
    parser.add_argument("--output", type=str, help="Output destination file path (.png or .tif)")
    args = parser.parse_args()

    if args.convert:
        if not args.output:
            parser.error("--output is required when using --convert")
        convert_pds4_img_to_raster(args.convert, args.output)
    else:
        print_issdc_download_guide()
