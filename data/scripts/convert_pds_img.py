"""Chandrayaan-2 PDS-4 .IMG to GeoTIFF/PNG Converter.

Demonstrates conversion of raw/calibrated Chandrayaan-2 OHRC and TMC-2 raster
products (.IMG / PDS-4) downloaded from ISSDC PRADAN into standard formats
(GeoTIFF, PNG, JPEG) expected by the SatQuery AI pipeline.

Usage:
    python data/scripts/convert_pds_img.py --input path/to/ch2_product.IMG --output path/to/output.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def convert_pds_img_to_raster(
    input_path: str | Path,
    output_path: str | Path,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Path:
    """Convert a Chandrayaan-2 PDS-4 .IMG file into a normalized PNG/GeoTIFF."""
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(f"Input PDS-4 file not found: {in_file}")

    print(f"[PDS-4 Converter] Reading {in_file.name}...")

    # Method 1: Use rasterio if available (handles PDS driver and GeoTIFF georeferencing)
    if HAS_RASTERIO and out_file.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(in_file) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            meta = src.meta.copy()

        if nodata is not None:
            data[data == nodata] = np.nan

        # Contrast stretch
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            p_low = np.percentile(valid, percentile_low)
            p_high = np.percentile(valid, percentile_high)
            stretched = np.clip((data - p_low) / (p_high - p_low + 1e-6), 0, 1) * 255.0
            data_uint8 = np.nan_to_num(stretched, nan=0).astype(np.uint8)
        else:
            data_uint8 = np.zeros_like(data, dtype=np.uint8)

        meta.update({"driver": "GTiff", "dtype": "uint8", "count": 1})
        with rasterio.open(out_file, "w", **meta) as dst:
            dst.write(data_uint8, 1)

        print(f"[PDS-4 Converter] Wrote GeoTIFF: {out_file}")
        return out_file

    # Method 2: Raw binary parsing or PIL read
    try:
        # Many calibrated OHRC/TMC-2 files are raw IEEE float32 or uint16 arrays
        raw_bytes = in_file.read_bytes()
        # Fallback to standard raster load if header is recognized
        try:
            pil_img = Image.open(in_file)
            arr = np.array(pil_img).astype(np.float32)
        except Exception:
            # Assume 16-bit or 32-bit square raster
            n_pix = len(raw_bytes) // 2
            dim = int(np.sqrt(n_pix))
            arr = np.frombuffer(raw_bytes[:dim*dim*2], dtype=np.uint16).reshape((dim, dim)).astype(np.float32)

        # Percentile contrast stretch for display
        p_low = np.percentile(arr, percentile_low)
        p_high = np.percentile(arr, percentile_high)
        scaled = np.clip((arr - p_low) / (p_high - p_low + 1e-6) * 255.0, 0, 255).astype(np.uint8)

        out_img = Image.fromarray(scaled)
        out_img.save(out_file)
        print(f"[PDS-4 Converter] Successfully exported {out_file} ({out_img.size})")
        return out_file

    except Exception as exc:
        print(f"[PDS-4 Converter] Error converting {in_file}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Chandrayaan-2 PDS-4 .IMG to PNG/GeoTIFF.")
    parser.add_argument("--input", "-i", required=True, help="Input .IMG file path")
    parser.add_argument("--output", "-o", required=True, help="Output .png or .tif file path")
    args = parser.parse_args()

    convert_pds_img_to_raster(args.input, args.output)
