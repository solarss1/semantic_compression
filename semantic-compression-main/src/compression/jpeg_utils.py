"""Спільні операції JPEG для класичного та семантичного стиснення."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def jpeg_encode(image: np.ndarray, quality: int, *, optimize: bool = True) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(
        buf, format="JPEG", quality=quality, optimize=optimize
    )
    return buf.getvalue()


def jpeg_decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def jpeg_roundtrip(image: np.ndarray, quality: int) -> tuple[np.ndarray, bytes]:
    data = jpeg_encode(image, quality)
    return jpeg_decode(data), data


def blended_jpeg_quality(
    roi_mask: np.ndarray,
    quality_background: int,
    quality_roi: int,
) -> int:
    """Якість одного фінального JPEG за середньою вагою ROI на кадрі."""
    weight = float(np.clip(roi_mask, 0.0, 1.0).mean())
    q = int(quality_background + weight * (quality_roi - quality_background))
    return max(1, min(95, q))
