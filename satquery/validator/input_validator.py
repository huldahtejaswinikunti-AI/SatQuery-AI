"""
Input validator implementation.
"""
from __future__ import annotations
import numpy as np
from satquery.validator.schemas import Modality, InputImageMetadata, ValidationResult

class InputValidator:
    @staticmethod
    def infer_modality(arr: np.ndarray, meta: dict) -> Modality:
        channels = arr.shape[-1] if arr.ndim == 3 else 1
        name = str(meta.get("filename", "")).lower()
        if any(x in name for x in ["sar", "s1", "vv", "vh"]):
            return Modality.SAR
        if "s2" in name or "optical" in name:
            return Modality.MULTISPECTRAL if channels >= 4 else Modality.OPTICAL
        if channels in (1, 2):
            return Modality.SAR
        elif channels == 3:
            return Modality.OPTICAL
        elif channels >= 4:
            return Modality.MULTISPECTRAL
        return Modality.OPTICAL

    def validate(self, images: list[np.ndarray], metas: list[dict], query: str = "", user_declared_task: str | None = None) -> ValidationResult:
        warnings = []
        if len(images) == 0:
            return ValidationResult(is_valid=False, num_images=0, error_message="No images provided.")
        if len(images) > 2:
            return ValidationResult(is_valid=False, num_images=len(images), error_message="Maximum of 2 images supported.")

        img_metas = []
        for i, (arr, m) in enumerate(zip(images, metas)):
            h, w = arr.shape[:2]
            c = arr.shape[-1] if arr.ndim == 3 else 1
            mod = self.infer_modality(arr, m)
            img_metas.append(InputImageMetadata(
                filename=m.get("filename", f"image_{i+1}"),
                width=w, height=h, channels=c, modality=mod,
                format=m.get("format", "UNKNOWN"), crs=m.get("crs"),
            ))

        if len(images) == 1:
            return ValidationResult(is_valid=True, num_images=1, images_metadata=img_metas, detected_configuration="single_image")

        modalities = {m.modality for m in img_metas}
        has_sar = Modality.SAR in modalities
        has_opt = (Modality.OPTICAL in modalities) or (Modality.MULTISPECTRAL in modalities)
        detected = "optical_sar_pair" if (has_sar and has_opt) else "bitemporal_pair"
        return ValidationResult(is_valid=True, num_images=2, images_metadata=img_metas, detected_configuration=detected, warnings=warnings)
