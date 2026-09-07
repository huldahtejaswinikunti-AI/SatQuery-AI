"""Tests for satquery.validator.input_validator."""

from __future__ import annotations

import pytest

from satquery.validator.input_validator import validate_input
from satquery.validator.schemas import ValidatedInput, ValidationError, InputType


class TestSingleImage:
    """Single-image submissions."""

    def test_valid_optical_single(self, optical_image_meta):
        result = validate_input([optical_image_meta], "How many buildings?")
        assert isinstance(result, ValidatedInput)
        assert result.input_type == InputType.SINGLE
        assert len(result.images) == 1
        assert result.query == "How many buildings?"

    def test_valid_sar_single(self, sar_image_meta):
        result = validate_input([sar_image_meta], "Is there flooding?")
        assert isinstance(result, ValidatedInput)
        assert result.input_type == InputType.SINGLE


class TestPairs:
    """Two-image pair submissions."""

    def test_optical_sar_pair(self, optical_image_meta, sar_image_meta):
        result = validate_input(
            [optical_image_meta, sar_image_meta],
            "Fuse these images",
        )
        assert isinstance(result, ValidatedInput)
        assert result.input_type == InputType.OPTICAL_SAR_PAIR

    def test_bitemporal_pair(self, optical_image_meta):
        img1 = {**optical_image_meta, "timestamp": "2024-01-01T00:00:00+00:00"}
        img2 = {**optical_image_meta, "timestamp": "2025-01-01T00:00:00+00:00"}
        result = validate_input([img1, img2], "What changed?")
        assert isinstance(result, ValidatedInput)
        assert result.input_type == InputType.BITEMPORAL_PAIR


class TestValidationErrors:
    """Submissions that must fail with structured errors."""

    def test_no_images(self):
        result = validate_input([], "Hello")
        assert isinstance(result, ValidationError)
        assert result.code == "NO_IMAGES"

    def test_too_many_images(self, optical_image_meta):
        result = validate_input(
            [optical_image_meta, optical_image_meta, optical_image_meta],
            "Query",
        )
        assert isinstance(result, ValidationError)
        assert result.code == "IMAGE_COUNT_EXCEEDED"

    def test_empty_query(self, optical_image_meta):
        result = validate_input([optical_image_meta], "")
        assert isinstance(result, ValidationError)
        assert result.code == "EMPTY_QUERY"

    def test_crs_mismatch(self, optical_image_meta, optical_image_meta_different_crs):
        result = validate_input(
            [optical_image_meta, optical_image_meta_different_crs],
            "Compare",
        )
        assert isinstance(result, ValidationError)
        assert result.code == "CRS_MISMATCH"

    def test_bitemporal_missing_timestamps(self, optical_image_meta):
        img1 = {**optical_image_meta}
        img1.pop("timestamp", None)
        img2 = {**optical_image_meta}
        img2.pop("timestamp", None)
        result = validate_input([img1, img2], "What changed?")
        assert isinstance(result, ValidationError)
        assert result.code == "TIMESTAMP_REQUIRED"

    def test_unsupported_format(self, tmp_path):
        bad_image = {
            "path": str(tmp_path / "test.bmp"),
            "format": "BMP",
            "band_count": 3,
            "width": 64,
            "height": 64,
        }
        result = validate_input([bad_image], "Describe")
        assert isinstance(result, ValidationError)
        assert result.code == "UNSUPPORTED_FORMAT"
