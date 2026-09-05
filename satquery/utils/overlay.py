"""
Visual overlay generation.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

def create_mask_overlay(base_image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 69, 0), alpha: float = 0.5, threshold: float = 0.5) -> np.ndarray:
    if base_image.dtype != np.uint8:
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)
    h, w = base_image.shape[:2]
    if mask.shape[:2] != (h, w):
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = np.array(pil_mask.resize((w, h), Image.Resampling.BILINEAR)) / 255.0
    binary = mask >= threshold
    overlay = base_image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[binary] = (1.0 - alpha) * overlay[binary] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)

def create_change_overlay(before_rgb: np.ndarray, after_rgb: np.ndarray, change_mask: np.ndarray, color: tuple[int, int, int] = (255, 50, 50), alpha: float = 0.55) -> np.ndarray:
    return create_mask_overlay(after_rgb, change_mask, color, alpha)

def create_side_by_side(images: list[np.ndarray], titles: list[str] | None = None, padding: int = 8, bg_color: tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    target_h = max(img.shape[0] for img in images)
    resized_images = []
    for img in images:
        if img.shape[0] != target_h:
            h, w = img.shape[:2]
            new_w = int(w * (target_h / h))
            pil_img = Image.fromarray(img)
            img = np.array(pil_img.resize((new_w, target_h), Image.Resampling.BILINEAR))
        resized_images.append(img)
    total_w = sum(img.shape[1] for img in resized_images) + padding * (len(images) + 1)
    banner_h = 32 if titles else 0
    total_h = target_h + padding * 2 + banner_h
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    curr_x = padding
    for i, img in enumerate(resized_images):
        h, w = img.shape[:2]
        canvas[padding + banner_h : padding + banner_h + h, curr_x : curr_x + w] = img
        curr_x += w + padding
    if titles:
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        curr_x = padding
        for i, img in enumerate(resized_images):
            w = img.shape[1]
            if i < len(titles):
                draw.text((curr_x + 6, padding + 6), titles[i], fill=(240, 240, 240))
            curr_x += w + padding
        canvas = np.array(pil_canvas)
    return canvas
