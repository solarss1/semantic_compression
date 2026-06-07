"""Семантичне стиснення: адаптивна якість JPEG за маскою ROI."""

from __future__ import annotations

import numpy as np

from src.compression.jpeg_utils import (
    blended_jpeg_quality,
    jpeg_encode,
    jpeg_roundtrip,
)


class SemanticCompressor:
    """
    Адаптивне JPEG-стиснення з ROI.

    1. Базовий шар — одне JPEG-кодування всього кадру на ``quality_background``.
    2. У тайлах із помітним ROI — повторне кодування фрагмента з оригіналу
       на підвищеній якості (лише там, де це дає виграш).
    3. Розмір файлу для метрик — один фінальний JPEG відновленого зображення
       (порівнянно з класичним JPEG, а не сума сотень міні-JPEG).
    """

    def __init__(
        self,
        quality_roi: int = 85,
        quality_background: int = 35,
        tile_size: int = 32,
        roi_refine_threshold: float = 0.08,
    ) -> None:
        self.quality_roi = quality_roi
        self.quality_background = quality_background
        self.tile_size = tile_size
        self.roi_refine_threshold = roi_refine_threshold

    def compress(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
    ) -> tuple[np.ndarray, bytes, np.ndarray]:
        """
        Args:
            image: RGB (H, W, 3)
            roi_mask: float [0, 1], (H, W)

        Returns:
            (відновлене зображення, bitstream JPEG, карта якості)
        """
        h, w = image.shape[:2]
        if roi_mask.shape[:2] != (h, w):
            raise ValueError("ROI mask size must match image size")

        tile = self.tile_size
        base_restored, _ = jpeg_roundtrip(image, self.quality_background)
        output = base_restored.copy()
        quality_map = np.full((h, w), self.quality_background, dtype=np.int32)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y2 = min(y + tile, h)
                x2 = min(x + tile, w)
                mask_patch = roi_mask[y:y2, x:x2]
                weight = float(mask_patch.mean())
                q = int(
                    self.quality_background
                    + weight * (self.quality_roi - self.quality_background)
                )
                q = max(1, min(100, q))
                quality_map[y:y2, x:x2] = q

                if weight < self.roi_refine_threshold or q <= self.quality_background:
                    continue

                patch = image[y:y2, x:x2]
                restored, _ = jpeg_roundtrip(patch, q)
                output[y:y2, x:x2] = restored

        save_q = blended_jpeg_quality(
            roi_mask, self.quality_background, self.quality_roi
        )
        bitstream = jpeg_encode(output, save_q)
        return output, bitstream, quality_map
