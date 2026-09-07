"""Load raster and standard image files into a uniform dict representation.

Handles GeoTIFF/TIFF (via rasterio, with tifffile fallback) and plain
PNG/JPEG (via Pillow) for benchmark dataset samples.

Public API
----------
load_image(path) -> dict
    Returns ``{"bands": dict[str, ndarray], "metadata": dict}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from satquery.utils.config import (
    SUPPORTED_GEOTIFF_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def load_image(path: str) -> dict[str, Any]:
    """Load an image file and return bands + metadata.

    Parameters
    ----------
    path : str
        Absolute or relative filesystem path to the image.

    Returns
    -------
    dict
        ``{"bands": dict[str, np.ndarray], "metadata": dict}``

        For GeoTIFF:
            - bands: ``{"band_1": array, "band_2": array, ...}``
            - metadata keys: ``crs, transform, band_count, bounds, dtype,
              nodata, width, height, format``

        For PNG/JPEG:
            - bands: ``{"red": array, "green": array, "blue": array}``
              (+ ``"alpha"`` if present, or ``"gray"`` for grayscale)
            - metadata keys: ``crs=None, transform=None, band_count, width,
              height, format``

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    ValueError
        If the file extension is not in the supported set.
    """
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = filepath.suffix.lower()

    if ext in SUPPORTED_GEOTIFF_EXTENSIONS:
        return _load_geotiff(filepath)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        return _load_standard_image(filepath)
    else:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_GEOTIFF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# GeoTIFF / TIFF loader
# ---------------------------------------------------------------------------

def _load_geotiff(filepath: Path) -> dict[str, Any]:
    """Attempt rasterio first (full geo metadata); fall back to tifffile."""
    try:
        return _load_with_rasterio(filepath)
    except Exception:
        # File may be a plain TIFF without geo headers.
        return _load_with_tifffile(filepath)


def _load_with_rasterio(filepath: Path) -> dict[str, Any]:
    """Load via rasterio — extracts CRS, transform, bounds, etc."""
    import rasterio

    with rasterio.open(str(filepath)) as dataset:
        bands: dict[str, np.ndarray] = {}
        for idx in range(1, dataset.count + 1):
            bands[f"band_{idx}"] = dataset.read(idx)

        crs_value = None
        if dataset.crs is not None:
            try:
                crs_value = dataset.crs.to_epsg()
            except Exception:
                crs_value = dataset.crs.to_wkt()

        transform_tuple = None
        if dataset.transform is not None:
            transform_tuple = tuple(dataset.transform)[:6]

        bounds_dict = None
        if dataset.bounds is not None:
            b = dataset.bounds
            bounds_dict = {
                "left": b.left,
                "bottom": b.bottom,
                "right": b.right,
                "top": b.top,
            }

        metadata = {
            "crs": crs_value,
            "transform": transform_tuple,
            "band_count": dataset.count,
            "bounds": bounds_dict,
            "dtype": str(dataset.dtypes[0]),
            "nodata": dataset.nodata,
            "width": dataset.width,
            "height": dataset.height,
            "format": "GeoTIFF",
        }

    return {"bands": bands, "metadata": metadata}


def _load_with_tifffile(filepath: Path) -> dict[str, Any]:
    """Fallback for plain TIFF files without geospatial headers."""
    import tifffile

    data = tifffile.imread(str(filepath))

    bands: dict[str, np.ndarray] = {}

    if data.ndim == 2:
        # Single-band / grayscale
        bands["band_1"] = data
        band_count = 1
    elif data.ndim == 3:
        # (H, W, C) or (C, H, W) — heuristic: smallest dim is bands
        if data.shape[0] < data.shape[2]:
            # (C, H, W)
            for idx in range(data.shape[0]):
                bands[f"band_{idx + 1}"] = data[idx]
            band_count = data.shape[0]
        else:
            # (H, W, C)
            for idx in range(data.shape[2]):
                bands[f"band_{idx + 1}"] = data[:, :, idx]
            band_count = data.shape[2]
    else:
        raise ValueError(
            f"Unexpected TIFF array shape {data.shape}; expected 2-D or 3-D."
        )

    metadata = {
        "crs": None,
        "transform": None,
        "band_count": band_count,
        "bounds": None,
        "dtype": str(data.dtype),
        "nodata": None,
        "width": bands["band_1"].shape[1],
        "height": bands["band_1"].shape[0],
        "format": "TIFF",
    }

    return {"bands": bands, "metadata": metadata}


# ---------------------------------------------------------------------------
# PNG / JPEG loader
# ---------------------------------------------------------------------------

def _load_standard_image(filepath: Path) -> dict[str, Any]:
    """Load a standard image via Pillow."""
    from PIL import Image

    img = Image.open(str(filepath))
    arr = np.array(img)

    bands: dict[str, np.ndarray] = {}

    if arr.ndim == 2:
        # Grayscale
        bands["gray"] = arr
        band_count = 1
    elif arr.ndim == 3:
        channel_count = arr.shape[2]
        if channel_count == 1:
            bands["gray"] = arr[:, :, 0]
            band_count = 1
        elif channel_count == 3:
            bands["red"] = arr[:, :, 0]
            bands["green"] = arr[:, :, 1]
            bands["blue"] = arr[:, :, 2]
            band_count = 3
        elif channel_count == 4:
            bands["red"] = arr[:, :, 0]
            bands["green"] = arr[:, :, 1]
            bands["blue"] = arr[:, :, 2]
            bands["alpha"] = arr[:, :, 3]
            band_count = 4
        else:
            for idx in range(channel_count):
                bands[f"band_{idx + 1}"] = arr[:, :, idx]
            band_count = channel_count
    else:
        raise ValueError(
            f"Unexpected image array shape {arr.shape}; expected 2-D or 3-D."
        )

    fmt = filepath.suffix.lower()
    format_name = "PNG" if fmt == ".png" else "JPEG"

    metadata = {
        "crs": None,
        "transform": None,
        "band_count": band_count,
        "bounds": None,
        "dtype": str(arr.dtype),
        "nodata": None,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "format": format_name,
    }

    return {"bands": bands, "metadata": metadata}
