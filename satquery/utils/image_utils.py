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
