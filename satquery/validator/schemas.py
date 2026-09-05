"""
Validation schemas.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"

class ImageFormat(str, Enum):
    GEOTIFF = "GeoTIFF"
    TIFF = "TIFF"
    PNG = "PNG"
    JPEG = "JPEG"
    UNKNOWN = "UNKNOWN"

class InputImageMetadata(BaseModel):
    filename: Optional[str] = None
    width: int
    height: int
    channels: int
    modality: Modality = Modality.UNKNOWN
    format: str = "UNKNOWN"
    crs: Optional[str] = None
    transform: Optional[list[float]] = None
    timestamp: Optional[str] = None

class ValidationResult(BaseModel):
    is_valid: bool
    num_images: int
    images_metadata: list[InputImageMetadata] = Field(default_factory=list)
    detected_configuration: str = "single_image"
    error_message: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
