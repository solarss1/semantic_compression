"""Сервісний шар: ROI, стиснення, візуалізація."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.app.models import AppSettings, CompressionOutput
from src.compression.classical import (
    compress_jpeg,
    compress_png,
    compress_webp,
)
from src.compression.jpeg_utils import jpeg_decode
from src.compression.semantic import SemanticCompressor
from src.evaluation.metrics import evaluate_pair, raw_image_bytes
from src.roi.cache import extract_with_cache
from src.roi.factory import create_roi_extractor, release_roi_extractors
from src.roi.mask_utils import smooth_roi_mask
from src.roi.u2net_setup import prepare_roi_models
from src.roi.ultralytics_sam_setup import (
    parse_sam_boxes,
    parse_sam_points,
    scale_sam_prompts,
)
from src.utils.image_limits import limit_image_size, max_pixels_from_config


def extract_roi_mask(
    image: np.ndarray,
    config: dict[str, Any],
    settings: AppSettings,
    image_path: Path | None = None,
    sam_points: list[tuple[int, int, int]] | None = None,
    sam_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Виділити ROI-маску згідно з налаштуваннями."""
    cfg = settings.merge_into_config(config)
    if sam_points is not None or sam_boxes is not None:
        cfg.setdefault("roi", {})
    if sam_points is not None:
        cfg["roi"]["sam_points"] = [[x, y, label] for x, y, label in sam_points]
    if sam_boxes is not None:
        cfg["roi"]["sam_boxes"] = [list(box) for box in sam_boxes]
    prepare_roi_models(cfg, method=settings.roi_method, verbose=False)

    max_px = max_pixels_from_config(cfg)
    work = limit_image_size(image, max_px)

    if settings.roi_method == "ultralytics_sam" and (sam_points or sam_boxes):
        pts = parse_sam_points(cfg.get("roi", {}).get("sam_points"))
        boxes = parse_sam_boxes(cfg.get("roi", {}).get("sam_boxes"))
        pts, boxes = scale_sam_prompts(
            image.shape[:2], work.shape[:2], pts, boxes
        )
        cfg.setdefault("roi", {})
        cfg["roi"]["sam_points"] = [[x, y, label] for x, y, label in pts]
        cfg["roi"]["sam_boxes"] = [list(b) for b in boxes]

    cache = Path(cfg.get("app", {}).get("cache_dir", ".cache/roi"))

    release_roi_extractors()
    reuse = settings.roi_method not in ("combined", "ultralytics_sam")
    extractor = create_roi_extractor(cfg, reuse=reuse)
    try:
        if settings.roi_method == "ultralytics_sam":
            mask = extractor.extract(work)
        else:
            mask = extract_with_cache(
                extractor,
                work,
                image_path,
                settings.roi_method,
                cache,
            )
        mask = smooth_roi_mask(mask, settings.mask_smooth_sigma)
        if work.shape[:2] != image.shape[:2]:
            import cv2

            h, w = image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask.astype(np.float32)
    finally:
        release_roi_extractors()


def compress_classical(
    image: np.ndarray,
    settings: AppSettings,
) -> CompressionOutput:
    """Класичне стиснення (JPEG / WebP / PNG)."""
    method = settings.classical_method.lower()
    q = settings.classical_quality

    if method == "jpeg":
        result = compress_jpeg(image, q)
        ext = "jpg"
        label = f"JPEG (Q={q})"
    elif method == "webp":
        result = compress_webp(image, q)
        ext = "webp"
        label = f"WebP (Q={q})"
    elif method == "png":
        result = compress_png(image, settings.png_compress_level)
        ext = "png"
        label = f"PNG (рівень {settings.png_compress_level})"
    else:
        raise ValueError(f"Невідомий метод: {method}")

    metrics = evaluate_pair(image, result.reconstructed, result.compressed_bytes)
    orig = raw_image_bytes(image)
    return CompressionOutput(
        method_label=label,
        reconstructed=result.reconstructed,
        metrics=metrics,
        encode_ms=result.encode_ms,
        decode_ms=result.decode_ms,
        original_bytes=orig,
        bitstream=result.bitstream,
        file_extension=ext,
    )


def compress_semantic(
    image: np.ndarray,
    roi_mask: np.ndarray,
    settings: AppSettings,
) -> CompressionOutput:
    """Семантичне стиснення з адаптивною якістю за ROI."""
    if roi_mask.shape[:2] != image.shape[:2]:
        raise ValueError("Розмір маски ROI не збігається з зображенням")

    compressor = SemanticCompressor(
        quality_roi=settings.quality_roi,
        quality_background=settings.quality_background,
        tile_size=settings.tile_size,
    )
    import time

    t0 = time.perf_counter()
    reconstructed, bitstream, quality_map = compressor.compress(image, roi_mask)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    _ = jpeg_decode(bitstream)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    file_bytes = len(bitstream)
    orig = raw_image_bytes(image)
    metrics = evaluate_pair(image, reconstructed, file_bytes, roi_mask)

    return CompressionOutput(
        method_label=(
            f"Семантичне (ROI Q={settings.quality_roi}/"
            f"фон Q={settings.quality_background})"
        ),
        reconstructed=reconstructed,
        metrics=metrics,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        original_bytes=orig,
        bitstream=bitstream,
        file_extension="jpg",
        quality_map=quality_map,
    )


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Grayscale маска → RGB для відображення."""
    m = np.clip(mask, 0.0, 1.0)
    gray = (m * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def roi_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Накладення ROI на оригінал."""
    m = np.clip(mask[..., None], 0.0, 1.0)
    base = image.astype(np.float32)
    tint = np.zeros_like(base)
    tint[..., 0] = 255 * m[..., 0]
    tint[..., 1] = 180 * m[..., 0]
    out = base * (1.0 - alpha * m) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def sam_prompts_overlay(
    image: np.ndarray,
    points: list[tuple[int, int, int]] | None = None,
    boxes: list[tuple[int, int, int, int]] | None = None,
    *,
    draft_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Накласти SAM-кліки (+/−) та bbox на RGB-зображення."""
    import cv2

    out = image.copy()
    h, w = out.shape[:2]
    radius = max(3, min(h, w) // 250)
    thickness = 1
    font_scale = max(0.3, min(h, w) / 2500.0)
    box_thickness = max(2, min(h, w) // 400)

    def _draw_box_rgb(
        x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]
    ) -> None:
        xa, xb = sorted((max(0, x1), min(w - 1, x2)))
        ya, yb = sorted((max(0, y1), min(h - 1, y2)))
        if xb <= xa or yb <= ya:
            return
        t = box_thickness
        out[ya : ya + t, xa : xb + 1] = color
        out[yb - t + 1 : yb + 1, xa : xb + 1] = color
        out[ya : yb + 1, xa : xa + t] = color
        out[ya : yb + 1, xb - t + 1 : xb + 1] = color

    for x1, y1, x2, y2 in boxes or []:
        _draw_box_rgb(x1, y1, x2, y2, (255, 200, 0))

    if draft_box is not None:
        x1, y1, x2, y2 = draft_box
        _draw_box_rgb(x1, y1, x2, y2, (255, 240, 120))

    for x, y, label in points or []:
        color = (80, 220, 80) if label else (220, 80, 80)
        cv2.circle(out, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(out, (x, y), max(1, radius // 3), color, -1, cv2.LINE_AA)
        sign = "+" if label else "-"
        (tw, th), _ = cv2.getTextSize(sign, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(
            out,
            sign,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def compression_diff(
    original: np.ndarray,
    reconstructed: np.ndarray,
    amplify: float = 10.0,
) -> np.ndarray:
    """Підсилена різниця |оригінал − стиснене|."""
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    if diff.max() > 0:
        diff = np.clip(diff * amplify, 0, 255)
    return diff.astype(np.uint8)


def quality_map_rgb(quality_map: np.ndarray) -> np.ndarray:
    """Карта якості JPEG по тайлах → pseudo-RGB."""
    import cv2

    q = quality_map.astype(np.float32)
    q_norm = (q - q.min()) / (q.max() - q.min() + 1e-8)
    heat = (q_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
