"""Обмеження розміру зображень для стабільності DL-інференсу."""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_PIXELS = 768 * 512


def max_pixels_from_config(config: dict | None) -> int:
    if not config:
        return DEFAULT_MAX_PIXELS
    app = config.get("app", {})
    val = app.get("max_image_pixels")
    return int(val) if val is not None else DEFAULT_MAX_PIXELS


def limit_image_size(
    image: np.ndarray,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """Зменшити зображення, якщо H×W перевищує ліміт (зберігає пропорції)."""
    h, w = image.shape[:2]
    if h * w <= max_pixels:
        return image

    import cv2

    scale = (max_pixels / (h * w)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(
        np.uint8
    )
