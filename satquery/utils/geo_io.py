"""
GeoTIFF and standard image I/O for remote sensing data.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Union, BinaryIO
import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

def load_image_as_array(source: Union[str, Path, bytes, BinaryIO]) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "crs": None, "transform": None, "format": "UNKNOWN", "channels": 1, "dtype": "uint8"
    }
    if isinstance(source, (str, Path)):
        p = Path(source)
        metadata["filename"] = p.name
        suffix = p.suffix.lower()
        if HAS_TIFFFILE and suffix in (".tif", ".tiff", ".geotiff"):
            try:
                arr = tifffile.imread(source)
                metadata["format"] = "TIFF"
                arr = _ensure_channel_last(arr)
                metadata["channels"] = arr.shape[-1] if arr.ndim == 3 else 1
                metadata["dtype"] = str(arr.dtype)
                return arr, metadata
            except Exception:
                pass
        with Image.open(source) as pil_img:
            arr = np.array(pil_img)
            metadata["format"] = pil_img.format or suffix.replace(".", "").upper()
            arr = _ensure_channel_last(arr)
            metadata["channels"] = arr.shape[-1] if arr.ndim == 3 else 1
            metadata["dtype"] = str(arr.dtype)
            return arr, metadata

    elif isinstance(source, (bytes, io.BytesIO)) or hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        raw_bytes = source.read() if hasattr(source, "read") else source
        if HAS_TIFFFILE:
            try:
                arr = tifffile.imread(io.BytesIO(raw_bytes))
                metadata["format"] = "TIFF"
                arr = _ensure_channel_last(arr)
                metadata["channels"] = arr.shape[-1] if arr.ndim == 3 else 1
                metadata["dtype"] = str(arr.dtype)
                return arr, metadata
            except Exception:
                pass
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            arr = np.array(pil_img)
            metadata["format"] = pil_img.format or "IMAGE"
            arr = _ensure_channel_last(arr)
            metadata["channels"] = arr.shape[-1] if arr.ndim == 3 else 1
            metadata["dtype"] = str(arr.dtype)
            return arr, metadata

    raise ValueError(f"Unsupported source type: {type(source)}")

def _ensure_channel_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        return np.transpose(arr, (1, 2, 0))
    return arr
