"""Pure image-processing helpers used across perception and specialist layers.

All functions are pure (no side effects, no I/O, no model calls).
They operate on numpy arrays only.
"""

from __future__ import annotations

import numpy as np


def resize(
    array: np.ndarray,
    target_h: int,
    target_w: int,
    method: str = "bilinear",
) -> np.ndarray:
    """Resize a 2-D or 3-D array to ``(target_h, target_w)``.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape ``(H, W)`` or ``(C, H, W)``.
    target_h, target_w : int
        Desired spatial dimensions.
    method : str
        Interpolation method: ``"nearest"`` or ``"bilinear"`` (default).

    Returns
    -------
    np.ndarray
        Resized array preserving the original number of dimensions.
    """
    from PIL import Image

    pil_methods = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
    }
    interp = pil_methods.get(method, Image.BILINEAR)

    if array.ndim == 2:
        img = Image.fromarray(array)
        resized = img.resize((target_w, target_h), interp)
        return np.array(resized)
    elif array.ndim == 3:
        # (C, H, W) — resize each channel independently
        channels = []
        for c in range(array.shape[0]):
            img = Image.fromarray(array[c])
            resized = img.resize((target_w, target_h), interp)
            channels.append(np.array(resized))
        return np.stack(channels, axis=0)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {array.shape}")


def normalize(
    array: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize array values.

    Parameters
    ----------
    array : np.ndarray
        Input array (any shape).
    method : str
        One of:
        - ``"minmax"``: scale to [0, 1] range.
        - ``"standard"``: zero-mean, unit-variance.
        - ``"sentinel2_reflectance"``: divide by 10 000 (Sentinel-2 L2A convention).

    Returns
    -------
    np.ndarray
        Normalized array as float32.
    """
    arr = array.astype(np.float32)

    if method == "minmax":
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min == 0:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    elif method == "standard":
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std

    elif method == "sentinel2_reflectance":
        return arr / 10_000.0

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Supported: 'minmax', 'standard', 'sentinel2_reflectance'."
        )


def extract_bands(
    bands_dict: dict[str, np.ndarray],
    band_names: list[str],
) -> np.ndarray:
    """Stack selected bands into a ``(C, H, W)`` array.

    Parameters
    ----------
    bands_dict : dict[str, np.ndarray]
        Mapping of band name → 2-D array, as returned by ``geo_io.load_image``.
    band_names : list[str]
        Ordered list of band names to extract, e.g. ``["band_4", "band_3", "band_2"]``.

    Returns
    -------
    np.ndarray
        Stacked array of shape ``(len(band_names), H, W)``.

    Raises
    ------
    KeyError
        If a requested band name is not present in *bands_dict*.
    """
    arrays = []
    for name in band_names:
        if name not in bands_dict:
            available = sorted(bands_dict.keys())
            raise KeyError(
                f"Band '{name}' not found. Available bands: {available}"
            )
        arrays.append(bands_dict[name])
    return np.stack(arrays, axis=0)


def to_rgb_preview(
    bands_dict: dict[str, np.ndarray],
    r: str = "band_4",
    g: str = "band_3",
    b: str = "band_2",
) -> np.ndarray:
    """Create an 8-bit RGB preview image from selected bands.

    Default band mapping is Sentinel-2 true color (B4=Red, B3=Green, B2=Blue).
    Falls back to ``red/green/blue`` keys if the numbered bands are absent
    (e.g. when the source is a PNG/JPEG).

    Parameters
    ----------
    bands_dict : dict[str, np.ndarray]
        Band name → 2-D array mapping.
    r, g, b : str
        Keys for the red, green, and blue channels.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(H, W, 3)``.
    """
    # Graceful fallback for standard images
    if r not in bands_dict and "red" in bands_dict:
        r, g, b = "red", "green", "blue"

    rgb = extract_bands(bands_dict, [r, g, b])  # (3, H, W)
    rgb = rgb.astype(np.float32)

    # Per-channel 2nd/98th percentile stretch for visual clarity
    for i in range(3):
        channel = rgb[i]
        p2 = np.percentile(channel, 2)
        p98 = np.percentile(channel, 98)
        if p98 - p2 > 0:
            channel = (channel - p2) / (p98 - p2)
        else:
            channel = np.zeros_like(channel)
        rgb[i] = np.clip(channel, 0.0, 1.0)

    # (3, H, W) -> (H, W, 3), scale to uint8
    rgb = np.transpose(rgb, (1, 2, 0))
    return (rgb * 255).astype(np.uint8)


def to_display_rgb(array: np.ndarray) -> np.ndarray:
    """Convert any raw numpy array to a uint8 (H, W, 3) image for display.

    Accepts:
    - (H, W)       -- grayscale, replicated to 3 channels
    - (H, W, 1)    -- squeezed to grayscale
    - (H, W, 3)    -- RGB
    - (H, W, 4)    -- RGBA, alpha dropped
    - (H, W, C)    -- first 3 channels used
    - (C, H, W)    -- transposed to (H, W, C), then first 3 channels

    Float arrays are percentile-stretched to [0, 255].  Integer arrays are
    clipped to [0, 255].

    Parameters
    ----------
    array : np.ndarray
        Input image array in any of the shapes described above.

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(H, W, 3)``.
    """
    arr = np.asarray(array)

    # Squeeze (H, W, 1) -> (H, W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # (C, H, W) heuristic: if ndim==3 and first dim is small relative to
    # the other two, treat as channels-first
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    # Grayscale -> 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Take first 3 channels if more than 3
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    # Ensure float for stretching
    arr = arr.astype(np.float32)

    # Per-channel percentile stretch
    for i in range(min(arr.shape[2], 3)):
        ch = arr[:, :, i]
        p2 = np.percentile(ch, 2)
        p98 = np.percentile(ch, 98)
        if p98 - p2 > 0:
            arr[:, :, i] = (ch - p2) / (p98 - p2)
        else:
            arr[:, :, i] = 0.0

    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def extract_optical_bands(array: np.ndarray) -> dict[str, np.ndarray]:
    """Extract standard optical bands (red, green, blue, nir, swir) from an array.

    Handles 2D, 3D (H,W,C) or (C,H,W), and formats with 3, 4, or 5+ bands.
    """
    arr = np.asarray(array)

    # If channels-first, transpose to channels-last
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        val = arr.astype(np.float32)
        return {"red": val, "green": val, "blue": val, "nir": val, "swir": val}

    channels = arr.shape[2]
    if channels >= 5:
        # Expected Sentinel-2 / Landsat standard order: Red, Green, Blue, NIR, SWIR
        return {
            "red": arr[:, :, 0].astype(np.float32),
            "green": arr[:, :, 1].astype(np.float32),
            "blue": arr[:, :, 2].astype(np.float32),
            "nir": arr[:, :, 3].astype(np.float32),
            "swir": arr[:, :, 4].astype(np.float32),
        }
    elif channels == 4:
        # Red, Green, Blue, NIR
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        nir = arr[:, :, 3].astype(np.float32)
        # 4-band optical (RGB + NIR) lacks SWIR band (e.g. Sentinel-2 B11/B12)
        return {"red": r, "green": g, "blue": b, "nir": nir, "swir": None}
    else:
        # RGB (3 bands)
        r = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        b = arr[:, :, 2].astype(np.float32)
        # Synthetic NIR estimate from Green/Red
        nir = np.clip(1.5 * g - 0.5 * r, 0.0, None)
        # RGB imagery does not contain a physical SWIR band; do not alias to Green
        return {"red": r, "green": g, "blue": b, "nir": nir, "swir": None}


def extract_sar_bands(array: np.ndarray) -> dict[str, np.ndarray]:
    """Extract standard SAR polarization channels (VV, VH) from an array."""
    arr = np.asarray(array)

    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        vv = arr.astype(np.float32)
        vh = arr.astype(np.float32)
    elif arr.ndim == 3:
        vv = arr[:, :, 0].astype(np.float32)
        vh = arr[:, :, 1].astype(np.float32) if arr.shape[2] > 1 else arr[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D SAR array, got shape {arr.shape}")

    return {"vv": vv, "vh": vh}


