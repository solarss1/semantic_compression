"""Кеш масок ROI."""

import hashlib
from pathlib import Path

import numpy as np

from src.roi.base import ROIExtractor


def _cache_key(image_path: Path, method: str) -> str:
    stat = image_path.stat()
    payload = f"{image_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{method}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def cache_path(cache_dir: Path, image_path: Path, method: str) -> Path:
    key = _cache_key(image_path, method)
    return cache_dir / f"{image_path.stem}_{method}_{key}.npy"


def extract_with_cache(
    extractor: ROIExtractor,
    image: np.ndarray,
    image_path: Path | None,
    method: str,
    cache_dir: Path | None,
) -> np.ndarray:
    if cache_dir is None or image_path is None:
        return extractor.extract(image)

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, image_path, method)
    if path.exists():
        return np.load(path, allow_pickle=False).astype(np.float32)

    mask = extractor.extract(image)
    np.save(path, mask)
    return mask
