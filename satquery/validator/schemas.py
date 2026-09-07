"""Pydantic v2 schemas describing validated pipeline inputs.

These models are the canonical representation of a user submission after
it has been checked by ``input_validator.validate_input()``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    """Sensor modality of an image."""

    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class InputType(str, Enum):
    """Classification of the overall input configuration."""

    SINGLE = "single"
    OPTICAL_SAR_PAIR = "optical_sar_pair"
    BITEMPORAL_PAIR = "bitemporal_pair"


class ImageMeta(BaseModel):
    """Metadata for a single uploaded image."""

    path: str = Field(..., description="Filesystem path to the image file.")
    format: str = Field(
        ...,
        description="File format: 'GeoTIFF', 'TIFF', 'PNG', or 'JPEG'.",
    )
    band_count: int = Field(..., ge=1, description="Number of spectral bands.")
    modality: Modality = Field(
        default=Modality.UNKNOWN,
        description="Detected sensor modality.",
    )
    crs: Optional[int | str] = Field(
        default=None,
        description="Coordinate reference system (EPSG code or WKT).",
    )
    transform: Optional[tuple] = Field(
        default=None,
        description="Affine transform as a 6-tuple.",
    )
    bounds: Optional[dict] = Field(
        default=None,
        description="Spatial bounds: {left, bottom, right, top}.",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Acquisition timestamp (required for bi-temporal pairs).",
    )
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class ValidatedInput(BaseModel):
    """Fully validated pipeline input — safe to pass to the router/executor."""

    images: list[ImageMeta] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="One or two validated image descriptors.",
    )
    input_type: InputType = Field(
        ...,
        description="Classification of the image set.",
    )
    query: str = Field(
        ...,
        min_length=1,
        description="User's natural-language query.",
    )


class ValidationError(BaseModel):
    """Structured error returned when validation fails.

    Designed to be displayed directly in a UI — never a bare Python traceback.
    """

    code: str = Field(
        ...,
        description="Machine-readable error code, e.g. 'IMAGE_COUNT_EXCEEDED'.",
    )
    message: str = Field(
        ...,
        description="Human-readable explanation suitable for display in a UI.",
    )
