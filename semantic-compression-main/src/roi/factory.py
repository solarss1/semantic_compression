"""Фабрика ROI-екстракторів з контролем пам'яті."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from src.roi.base import ROIExtractor
from src.roi.segmentation import segmentation_extractor_from_config
from src.roi.u2net_saliency import create_saliency_extractor

_EXTRACTOR_CACHE: dict[str, ROIExtractor] = {}

SAM_ALIASES = {"ultralytics_sam", "sam_ultralytics", "sam"}
SALIENCY_ALIASES = {"saliency", "saliency_u2net", "saliency_resnet"}
SEGMENTATION_ALIASES = {"segmentation", "segmentation_weighted", "segmentation_binary"}


def _free_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def resolve_device(roi_cfg: dict[str, Any]) -> str:
    device = roi_cfg.get("device", "auto")
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return str(device)


def release_roi_extractors(method: str | None = None) -> None:
    global _EXTRACTOR_CACHE
    if method is None:
        _EXTRACTOR_CACHE.clear()
    else:
        keys = [k for k in _EXTRACTOR_CACHE if k.startswith(f"{method}:")]
        for k in keys:
            _EXTRACTOR_CACHE.pop(k, None)
    if method is None or method in SAM_ALIASES:
        try:
            from src.roi.ultralytics_sam_setup import clear_ultralytics_sam_cache

            clear_ultralytics_sam_cache()
        except ImportError:
            pass
    _free_torch_memory()


def _resolve_saliency_model(roi_cfg: dict[str, Any], method: str) -> str:
    if method == "saliency_resnet":
        return "resnet"
    if method == "saliency_u2net":
        return "u2net"
    return roi_cfg.get("saliency_model", "u2net")


def _resolve_segmentation_weighted(roi_cfg: dict[str, Any], method: str) -> bool:
    if method == "segmentation_binary":
        return False
    if method == "segmentation_weighted":
        return True
    return bool(roi_cfg.get("use_class_weights", True))


def create_roi_extractor(
    config: dict[str, Any],
    *,
    reuse: bool = True,
) -> ROIExtractor:
    roi_cfg = config.get("roi", {})
    method = roi_cfg.get("method", "saliency")
    device = resolve_device(roi_cfg)
    cache_key = f"{method}:{device}"

    if reuse and cache_key in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[cache_key]

    if method in SALIENCY_ALIASES:
        model_name = _resolve_saliency_model(roi_cfg, method)
        extractor: ROIExtractor = create_saliency_extractor(model_name, device=device)
    elif method in SEGMENTATION_ALIASES:
        weighted = _resolve_segmentation_weighted(roi_cfg, method)
        extractor = segmentation_extractor_from_config(
            roi_cfg, device, weighted=weighted
        )
    elif method == "combined":
        extractor = _CombinedExtractor(roi_cfg, device=device)
    elif method in SAM_ALIASES:
        from src.roi.ultralytics_sam_roi import UltralyticsSamROIExtractor

        return UltralyticsSamROIExtractor.from_config(roi_cfg)
    else:
        raise ValueError(
            f"Unknown ROI method: {method}. "
            f"Доступні: saliency, saliency_u2net, saliency_resnet, "
            f"segmentation, segmentation_weighted, segmentation_binary, "
            f"combined, ultralytics_sam"
        )

    if reuse and method != "combined":
        _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


class _CombinedExtractor(ROIExtractor):
    """U²-Net салієнтність + зважена семантична маска (max)."""

    def __init__(self, roi_cfg: dict[str, Any], device: str | None = None) -> None:
        self.roi_cfg = roi_cfg
        self.device = device

    def extract(self, image: np.ndarray) -> np.ndarray:
        sal_model = _resolve_saliency_model(self.roi_cfg, "saliency")
        sal = create_saliency_extractor(sal_model, device=self.device)
        try:
            m1 = sal.extract(image)
        finally:
            del sal
            _free_torch_memory()

        seg = segmentation_extractor_from_config(
            self.roi_cfg, self.device, weighted=True
        )
        try:
            m2 = seg.extract(image)
        finally:
            del seg
            _free_torch_memory()

        return np.maximum(m1, m2).astype(np.float32)
