"""
Visual overlay generation for grounding, change detection, and UI download.

Provides:
    - draw_overlay(): Primary API — semi-transparent mask + bbox + label on RGB
    - create_mask_overlay(): Legacy — NumPy-based mask blending
    - create_change_overlay(): Legacy — change mask on the 'after' image
    - create_side_by_side(): Legacy — multi-image comparison strip

All overlays are designed to look clean on a projector screen:
    - Moderate opacity (not too transparent, not too opaque)
    - High-contrast outlines (white glow + dark stroke)
    - Readable label font size with contrasting background pill
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Color palette for overlays — designed for projector visibility
# ---------------------------------------------------------------------------
OVERLAY_COLORS = {
    "default":    (0, 180, 255),    # Bright cyan-blue
    "water":      (30, 144, 255),   # Dodger blue
    "vegetation": (50, 205, 50),    # Lime green
    "urban":      (255, 140, 0),    # Dark orange
    "change":     (255, 50, 80),    # Bright red
}


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """Try to load a good font, fall back to default."""
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    # Fallback to default
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_overlay(
    base_image: Union[np.ndarray, Image.Image],
    mask: Optional[np.ndarray] = None,
    bbox: Optional[tuple[int, int, int, int]] = None,
    label: str = "",
    mask_color: tuple[int, int, int] = OVERLAY_COLORS["default"],
    mask_alpha: float = 0.45,
    bbox_color: tuple[int, int, int] = (255, 255, 0),
    bbox_width: int = 3,
    label_font_size: int = 20,
) -> Image.Image:
    """
    Render a semi-transparent colored mask and/or bounding box with text
    label on top of a base RGB image.

    Designed for projector-screen readability: moderate opacity,
    high-contrast outline, and readable label with background pill.

    Parameters
    ----------
    base_image : np.ndarray (H, W, 3) uint8 or PIL.Image
        The base RGB image to overlay on.
    mask : np.ndarray (H, W), optional
        Boolean or float mask. If float, values > 0.5 are treated as positive.
        If None, no mask overlay is drawn.
    bbox : tuple (x1, y1, x2, y2), optional
        Bounding box in pixel coordinates. If None, no bbox is drawn.
    label : str
        Text label to draw. Placed near the bbox top-left or top-left of
        the mask centroid.
    mask_color : tuple (R, G, B)
        Color for the mask overlay. Default: bright cyan.
    mask_alpha : float
        Opacity of the mask overlay (0 = transparent, 1 = opaque). Default: 0.45.
    bbox_color : tuple (R, G, B)
        Color for the bounding box lines. Default: yellow.
    bbox_width : int
        Width of the bounding box lines. Default: 3.
    label_font_size : int
        Font size for the label text. Default: 20.

    Returns
    -------
    PIL.Image.Image
        The composited image with overlay(s) applied.
    """
    # Convert to PIL if needed
    if isinstance(base_image, np.ndarray):
        if base_image.dtype != np.uint8:
            base_image = np.clip(base_image, 0, 255).astype(np.uint8)
        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)
        pil_base = Image.fromarray(base_image[..., :3], "RGB")
    else:
        pil_base = base_image.convert("RGB")

    w, h = pil_base.size
    result = pil_base.copy()

    # ----- Mask overlay -----
    if mask is not None:
        # Ensure mask matches image dimensions
        if mask.shape[:2] != (h, w):
            mask_pil = Image.fromarray(
                (mask.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
            )
            mask = np.array(
                mask_pil.resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0

        # Binarize float masks
        if mask.dtype in (np.float32, np.float64):
            binary = mask >= 0.5
        else:
            binary = mask.astype(bool)

        # Create the colored mask layer
        mask_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mask_arr = np.array(mask_layer)

        # Fill positive region with the overlay color + alpha
        alpha_val = int(mask_alpha * 255)
        mask_arr[binary] = [mask_color[0], mask_color[1], mask_color[2], alpha_val]

        mask_layer = Image.fromarray(mask_arr, "RGBA")

        # Composite
        result = result.convert("RGBA")
        result = Image.alpha_composite(result, mask_layer)
        result = result.convert("RGB")

        # Draw high-contrast mask boundary outline
        try:
            from scipy import ndimage
            dilated = ndimage.binary_dilation(binary, iterations=2)
            boundary = dilated ^ binary
        except ImportError:
            from PIL import ImageFilter
            mask_img = Image.fromarray((binary * 255).astype(np.uint8))
            dilated_img = mask_img.filter(ImageFilter.MaxFilter(size=5))
            boundary = (np.array(dilated_img) > 127) ^ binary

        if np.any(boundary):
            draw = ImageDraw.Draw(result)
            ys, xs = np.where(boundary)
            for y, x in zip(ys.tolist(), xs.tolist()):
                # White glow (2px) + dark stroke (1px inner)
                draw.point((x, y), fill=(255, 255, 255))

    # ----- Bounding box -----
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(result)

        # Outer glow (white, wider)
        draw.rectangle(
            [x1 - 1, y1 - 1, x2 + 1, y2 + 1],
            outline=(255, 255, 255),
            width=bbox_width + 2,
        )
        # Main box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=bbox_color,
            width=bbox_width,
        )

    # ----- Label -----
    if label:
        draw = ImageDraw.Draw(result)
        font = _get_font(label_font_size)

        # Determine label position
        if bbox is not None:
            lx = bbox[0]
            ly = max(0, bbox[1] - label_font_size - 10)
        elif mask is not None and np.any(mask >= 0.5 if mask.dtype in (np.float32, np.float64) else mask):
            binary_for_pos = mask >= 0.5 if mask.dtype in (np.float32, np.float64) else mask.astype(bool)
            ys, xs = np.where(binary_for_pos)
            lx = max(0, int(np.min(xs)))
            ly = max(0, int(np.min(ys)) - label_font_size - 10)
        else:
            lx, ly = 10, 10

        # Measure text
        text_bbox = draw.textbbox((lx, ly), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # Background pill — dark with slight transparency
        pill_padding = 6
        pill_coords = [
            lx - pill_padding,
            ly - pill_padding,
            lx + tw + pill_padding,
            ly + th + pill_padding,
        ]
        # Clamp to image bounds
        pill_coords[0] = max(0, pill_coords[0])
        pill_coords[1] = max(0, pill_coords[1])
        pill_coords[2] = min(w, pill_coords[2])
        pill_coords[3] = min(h, pill_coords[3])

        # Draw background pill
        pill_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pill_draw = ImageDraw.Draw(pill_layer)
        pill_draw.rounded_rectangle(
            pill_coords, radius=4, fill=(0, 0, 0, 180),
        )
        result_rgba = result.convert("RGBA")
        result_rgba = Image.alpha_composite(result_rgba, pill_layer)
        result = result_rgba.convert("RGB")

        # Draw text (white, high contrast)
        draw = ImageDraw.Draw(result)
        # Shadow for extra readability
        draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0), font=font)
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

    return result


# ---------------------------------------------------------------------------
# Legacy API — kept intact for backward compatibility
# ---------------------------------------------------------------------------

def create_mask_overlay(
    base_image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 69, 0),
    alpha: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Blend a colored mask onto a base image. Returns np.ndarray (H, W, 3) uint8."""
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


def create_change_overlay(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    change_mask: np.ndarray,
    color: tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay a change mask on the 'after' image."""
    return create_mask_overlay(after_rgb, change_mask, color, alpha)


def create_side_by_side(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    padding: int = 8,
    bg_color: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """Create a horizontal strip of multiple images, optionally titled."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    target_h = max(img.shape[0] for img in images)
    resized_images = []
    for img in images:
        if img.shape[0] != target_h:
            h_img, w_img = img.shape[:2]
            new_w = int(w_img * (target_h / h_img))
            pil_img = Image.fromarray(img)
            img = np.array(pil_img.resize((new_w, target_h), Image.Resampling.BILINEAR))
        resized_images.append(img)
    total_w = sum(img.shape[1] for img in resized_images) + padding * (len(images) + 1)
    banner_h = 32 if titles else 0
    total_h = target_h + padding * 2 + banner_h
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    curr_x = padding
    for i, img in enumerate(resized_images):
        h_img, w_img = img.shape[:2]
        canvas[padding + banner_h : padding + banner_h + h_img, curr_x : curr_x + w_img] = img
        curr_x += w_img + padding
    if titles:
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)
        font = _get_font(14)
        curr_x = padding
        for i, img in enumerate(resized_images):
            w_img = img.shape[1]
            if i < len(titles):
                draw.text((curr_x + 6, padding + 6), titles[i], fill=(240, 240, 240), font=font)
            curr_x += w_img + padding
        canvas = np.array(pil_canvas)
    return canvas
