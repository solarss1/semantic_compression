"""Моделі даних для desktop-додатку."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import (
    CompressionMetrics,
    compression_ratio,
    raw_image_bytes,
)


@dataclass
class AppSettings:
    """Налаштування стиснення та ROI (з UI або config)."""

    roi_method: str = "saliency_u2net"
    device: str = "cpu"
    mask_smooth_sigma: float = 5.0
    sam_model: str = "mobile_sam.pt"
    quality_roi: int = 85
    quality_background: int = 35
    tile_size: int = 32
    classical_method: str = "jpeg"
    classical_quality: int = 50
    png_compress_level: int = 6

    def merge_into_config(self, base: dict[str, Any]) -> dict[str, Any]:
        roi = {
            **base.get("roi", {}),
            "method": self.roi_method,
            "device": self.device,
            "mask_smooth_sigma": self.mask_smooth_sigma,
            "sam_model": self.sam_model,
        }
        compression = {
            **base.get("compression", {}),
            "quality_roi": self.quality_roi,
            "quality_background": self.quality_background,
            "tile_size": self.tile_size,
            "quality_uniform": self.classical_quality,
            "png_compress_level": self.png_compress_level,
        }
        return {**base, "roi": roi, "compression": compression}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AppSettings:
        roi = config.get("roi", {})
        comp = config.get("compression", {})
        return cls(
            roi_method=str(roi.get("method", "saliency_u2net")),
            device=str(roi.get("device", "cpu")),
            mask_smooth_sigma=float(roi.get("mask_smooth_sigma", 5.0)),
            sam_model=str(roi.get("sam_model", "mobile_sam.pt")),
            quality_roi=int(comp.get("quality_roi", 85)),
            quality_background=int(comp.get("quality_background", 35)),
            tile_size=int(comp.get("tile_size", 32)),
            classical_quality=int(
                comp.get("quality_uniform", comp.get("webp_quality", 50))
            ),
            png_compress_level=int(comp.get("png_compress_level", 6)),
        )


@dataclass
class CompressionOutput:
    """Результат одного методу стиснення."""

    method_label: str
    reconstructed: np.ndarray
    metrics: CompressionMetrics
    encode_ms: float
    decode_ms: float
    original_bytes: int
    bitstream: bytes
    file_extension: str
    quality_map: np.ndarray | None = None

    @property
    def compression_ratio(self) -> float:
        return compression_ratio(self.original_bytes, self.metrics.file_size_bytes)


@dataclass
class SessionState:
    """Поточна сесія користувача."""

    original: np.ndarray | None = None
    source_path: Path | None = None
    source_file_bytes: int | None = None
    roi_mask: np.ndarray | None = None
    classical: CompressionOutput | None = None
    semantic: CompressionOutput | None = None
    settings: AppSettings = field(default_factory=AppSettings)
    sam_points: list[tuple[int, int, int]] = field(default_factory=list)
    sam_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def has_image(self) -> bool:
        return self.original is not None

    @property
    def original_bytes(self) -> int:
        if self.original is None:
            return 0
        return raw_image_bytes(self.original)
