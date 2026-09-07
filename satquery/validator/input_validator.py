"""Input validation for the SatQuery AI pipeline.

Checks image count, format, modality consistency, co-registration for
pairs, and timestamp presence for bi-temporal pairs.  Returns either a
``ValidatedInput`` on success or a ``ValidationError`` on failure —
never a bare Python exception.

Public API
----------
validate_input(images, query) -> ValidatedInput | ValidationError
"""

from __future__ import annotations

from typing import Any

from satquery.utils.config import (
    ALL_SUPPORTED_EXTENSIONS,
    OPTICAL_BAND_COUNT_MIN,
    SAR_BAND_COUNT_MAX,
    SUPPORTED_GEOTIFF_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from satquery.validator.schemas import (
    ImageMeta,
    InputType,
    Modality,
    ValidatedInput,
    ValidationError,
)


def validate_input(
    images: list[dict[str, Any]],
    query: str,
) -> ValidatedInput | ValidationError:
    """Validate a user submission and return a typed result.

    Parameters
    ----------
    images : list[dict]
        Each dict must contain at least ``path`` and the metadata keys
        produced by ``geo_io.load_image()["metadata"]``.
    query : str
        The user's natural-language question.

    Returns
    -------
    ValidatedInput | ValidationError
        A validated input object or a structured, UI-displayable error.
    """
    # --- query check ---
    if not query or not query.strip():
        return ValidationError(
            code="EMPTY_QUERY",
            message="A query is required. Please enter a question about the image(s).",
        )

    # --- image count ---
    if not images:
        return ValidationError(
            code="NO_IMAGES",
            message="At least one image is required.",
        )
    if len(images) > 2:
        return ValidationError(
            code="IMAGE_COUNT_EXCEEDED",
            message=(
                f"At most 2 images are supported, but {len(images)} were provided. "
                "Please upload a single image or a pair (optical+SAR or bi-temporal)."
            ),
        )

    # --- per-image checks ---
    metas: list[ImageMeta] = []
    for idx, img in enumerate(images, start=1):
        result = _validate_single_image(img, idx)
        if isinstance(result, ValidationError):
            return result
        metas.append(result)

    # --- pair checks ---
    if len(metas) == 2:
        pair_error = _validate_pair(metas)
        if pair_error is not None:
            return pair_error

    # --- determine input type ---
    input_type = _classify_input(metas)

    return ValidatedInput(
        images=metas,
        input_type=input_type,
        query=query.strip(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_modality(band_count: int) -> Modality:
    """Heuristic: ≤ SAR_BAND_COUNT_MAX → SAR, ≥ OPTICAL_BAND_COUNT_MIN → optical."""
    if band_count <= SAR_BAND_COUNT_MAX:
        return Modality.SAR
    if band_count >= OPTICAL_BAND_COUNT_MIN:
        return Modality.OPTICAL
    return Modality.UNKNOWN


def _validate_single_image(
    img: dict[str, Any],
    index: int,
) -> ImageMeta | ValidationError:
    """Build and validate an ``ImageMeta`` from a raw dict."""
    path = img.get("path")
    if not path:
        return ValidationError(
            code="MISSING_PATH",
            message=f"Image {index}: file path is missing.",
        )

    # Format check
    fmt = img.get("format", "")
    ext_lower = ""
    if path:
        from pathlib import Path as _P
        ext_lower = _P(path).suffix.lower()

    if ext_lower and ext_lower not in ALL_SUPPORTED_EXTENSIONS:
        return ValidationError(
            code="UNSUPPORTED_FORMAT",
            message=(
                f"Image {index}: format '{ext_lower}' is not supported. "
                f"Accepted formats: {sorted(ALL_SUPPORTED_EXTENSIONS)}."
            ),
        )

    # For PNG/JPEG, only allowed as benchmark samples (we accept them but note it)
    band_count = img.get("band_count", 0)
    if band_count < 1:
        return ValidationError(
            code="INVALID_BAND_COUNT",
            message=f"Image {index}: band count must be ≥ 1, got {band_count}.",
        )

    modality = _detect_modality(band_count)

    return ImageMeta(
        path=path,
        format=fmt or ("GeoTIFF" if ext_lower in SUPPORTED_GEOTIFF_EXTENSIONS else
                        "PNG" if ext_lower == ".png" else
                        "JPEG" if ext_lower in {".jpg", ".jpeg"} else "UNKNOWN"),
        band_count=band_count,
        modality=modality,
        crs=img.get("crs"),
        transform=tuple(img["transform"]) if img.get("transform") else None,
        bounds=img.get("bounds"),
        timestamp=img.get("timestamp"),
        width=img.get("width", 0) or 1,
        height=img.get("height", 0) or 1,
    )


def _validate_pair(metas: list[ImageMeta]) -> ValidationError | None:
    """Cross-image checks for a two-image submission."""
    a, b = metas

    # CRS consistency (both must be present and equal for geo-referenced pairs)
    if a.crs is not None and b.crs is not None:
        if str(a.crs) != str(b.crs):
            return ValidationError(
                code="CRS_MISMATCH",
                message=(
                    f"Image pair CRS mismatch: image 1 has CRS '{a.crs}' "
                    f"but image 2 has CRS '{b.crs}'. Both images must share "
                    "the same coordinate reference system."
                ),
            )

    # Bounds overlap check
    if a.bounds is not None and b.bounds is not None:
        if not _bounds_overlap(a.bounds, b.bounds):
            return ValidationError(
                code="BOUNDS_NO_OVERLAP",
                message=(
                    "Image pair has no spatial overlap. For paired analysis, "
                    "both images must cover at least a partially overlapping area."
                ),
            )

    # Bi-temporal: if both are same modality, timestamps are required
    if a.modality == b.modality:
        if a.timestamp is None or b.timestamp is None:
            return ValidationError(
                code="TIMESTAMP_REQUIRED",
                message=(
                    "Both images have the same modality, indicating a bi-temporal pair. "
                    "Each image must include an acquisition timestamp."
                ),
            )

    return None


def _bounds_overlap(a: dict, b: dict) -> bool:
    """Return True if two bounding boxes have any spatial overlap."""
    return not (
        a["right"] < b["left"]
        or b["right"] < a["left"]
        or a["top"] < b["bottom"]
        or b["top"] < a["bottom"]
    )


def _classify_input(metas: list[ImageMeta]) -> InputType:
    """Determine the ``InputType`` from validated image metadata."""
    if len(metas) == 1:
        return InputType.SINGLE

    a, b = metas
    modalities = {a.modality, b.modality}

    if Modality.OPTICAL in modalities and Modality.SAR in modalities:
        return InputType.OPTICAL_SAR_PAIR

    # Same modality with timestamps → bi-temporal
    return InputType.BITEMPORAL_PAIR
