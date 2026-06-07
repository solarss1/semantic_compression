"""Ultralytics SAM — завантаження моделі та парсинг промптів."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import project_root
from src.utils.file_lock import exclusive_file_lock

ROOT = Path(__file__).resolve().parents[2]
SAM_DIR = ROOT / "checkpoints" / "sam"
SAM_LOCK_FILE = ROOT / "checkpoints" / ".sam_setup.lock"

DEFAULT_SAM_MODEL = "mobile_sam.pt"
SAM_MODEL_CHOICES = ("mobile_sam.pt", "sam_b.pt", "sam_l.pt")

MIN_SAM_BYTES: dict[str, int] = {
    "mobile_sam.pt": 5_000_000,
    "sam_b.pt": 100_000_000,
    "sam_l.pt": 300_000_000,
}
DEFAULT_MIN_SAM_BYTES = 1_000_000


class UltralyticsSamNotAvailableError(RuntimeError):
    pass


def ultralytics_import_error() -> str | None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return (
            "Пакет ultralytics не встановлено. "
            "Виконайте: make setup-sam або .venv/bin/pip install ultralytics"
        )
    return None


def sam_model_basename(model_name: str) -> str:
    name = Path(model_name).name.strip()
    if not name:
        raise ValueError("Порожня назва SAM-моделі")
    return name


def sam_model_path(model_name: str) -> Path:
    """Локальний шлях до ваг SAM у checkpoints/sam/."""
    name = sam_model_basename(model_name)
    return SAM_DIR / name


def sam_weights_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    min_bytes = MIN_SAM_BYTES.get(path.name, DEFAULT_MIN_SAM_BYTES)
    return path.stat().st_size >= min_bytes


def _migrate_existing_weights(name: str, dest: Path) -> None:
    """Перенести ваги з кореня проєкту або каталогу Ultralytics, якщо є."""
    candidates = [ROOT / name]
    try:
        from ultralytics.utils import SETTINGS

        candidates.append(Path(SETTINGS["weights_dir"]) / name)
    except ImportError:
        pass

    for src in candidates:
        if src.resolve() == dest.resolve():
            continue
        if src.is_file() and src.stat().st_size >= MIN_SAM_BYTES.get(
            name, DEFAULT_MIN_SAM_BYTES
        ):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            return


def ensure_sam_weights(
    model_name: str = DEFAULT_SAM_MODEL,
    *,
    verbose: bool = True,
) -> Path:
    """Завантажити ваги SAM у checkpoints/sam/ (якщо ще немає)."""
    err = ultralytics_import_error()
    if err:
        raise UltralyticsSamNotAvailableError(err)

    name = sam_model_basename(model_name)
    if name not in SAM_MODEL_CHOICES:
        raise ValueError(
            f"Невідома SAM-модель: {name}. Доступні: {', '.join(SAM_MODEL_CHOICES)}"
        )

    dest = sam_model_path(name)
    if sam_weights_ready(dest):
        return dest

    SAM_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_existing_weights(name, dest)
    if sam_weights_ready(dest):
        if verbose:
            print(f"  SAM (перенесено): {dest}")
        return dest

    with exclusive_file_lock(SAM_LOCK_FILE):
        if sam_weights_ready(dest):
            return dest
        _migrate_existing_weights(name, dest)
        if sam_weights_ready(dest):
            return dest

        if verbose:
            print(f"Завантаження {name} → {dest} …")

        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(dest))

    if not sam_weights_ready(dest):
        raise RuntimeError(f"Не вдалося завантажити SAM: {dest}")

    if verbose:
        print(f"  SAM: {dest}")
    return dest


def resolve_sam_model_name(roi_cfg: dict[str, Any] | None) -> str:
    if not roi_cfg:
        return DEFAULT_SAM_MODEL
    name = str(roi_cfg.get("sam_model", DEFAULT_SAM_MODEL)).strip()
    return name or DEFAULT_SAM_MODEL


def resolve_sam_model_path(model_name: str) -> Path:
    """Повний шлях до .pt у checkpoints/sam/ (без завантаження)."""
    return sam_model_path(resolve_sam_model_name({"sam_model": model_name}))


def parse_sam_points(raw: list[Any] | None) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    if not raw:
        return points
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            x, y, label = int(item[0]), int(item[1]), int(item[2])
            points.append((x, y, 1 if label else 0))
    return points


def parse_sam_boxes(raw: list[Any] | None) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not raw:
        return boxes
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            x1, y1, x2, y2 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def scale_sam_prompts(
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    points: list[tuple[int, int, int]],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """Масштабувати кліки/bbox з координат оригіналу до work-зображення."""
    sh, sw = src_shape
    dh, dw = dst_shape
    if (sh, sw) == (dh, dw):
        return list(points), list(boxes)

    sx, sy = dw / sw, dh / sh
    scaled_points = [
        (max(0, min(dw - 1, int(round(x * sx)))), max(0, min(dh - 1, int(round(y * sy)))), label)
        for x, y, label in points
    ]
    scaled_boxes = [
        (
            max(0, min(dw - 1, int(round(x1 * sx)))),
            max(0, min(dh - 1, int(round(y1 * sy)))),
            max(0, min(dw - 1, int(round(x2 * sx)))),
            max(0, min(dh - 1, int(round(y2 * sy)))),
        )
        for x1, y1, x2, y2 in boxes
    ]
    return scaled_points, scaled_boxes


def format_sam_point_prompts(
    points: list[tuple[int, int, int]],
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Ultralytics: один об'єкт, N точок — [[(x,y),…]], [[labels…]]."""
    if not points:
        return [], []
    coords = [[x, y] for x, y, _ in points]
    labels = [label for _, _, label in points]
    return [coords], [labels]


def masks_to_float32(masks: Any, height: int, width: int) -> np.ndarray:
    if masks is None:
        return np.zeros((height, width), dtype=np.float32)
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    merged = np.zeros((height, width), dtype=np.float32)
    for m in arr:
        m = np.asarray(m, dtype=np.float32)
        if m.shape != (height, width):
            import cv2

            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        merged = np.maximum(merged, m)
    return np.clip(merged, 0.0, 1.0)


@lru_cache(maxsize=4)
def _load_sam_model(resolved_path: str, device: str) -> Any:
    err = ultralytics_import_error()
    if err:
        raise UltralyticsSamNotAvailableError(err)
    from ultralytics import SAM

    return SAM(resolved_path)


def get_sam_model(model_name: str, device: str) -> Any:
    weights = ensure_sam_weights(model_name, verbose=False)
    return _load_sam_model(str(weights.resolve()), device)


def clear_ultralytics_sam_cache() -> None:
    _load_sam_model.cache_clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def prepare_sam_model(
    model_name: str = DEFAULT_SAM_MODEL,
    *,
    verbose: bool = True,
) -> Path:
    """Завантажити ваги SAM у checkpoints/sam/ і перевірити завантаження моделі."""
    weights = ensure_sam_weights(model_name, verbose=verbose)
    if verbose:
        print(f"Ultralytics SAM: {weights.name} …")
    get_sam_model(model_name, "cpu")
    if verbose:
        print("SAM готовий.")
    return weights


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Підготовка Ultralytics SAM")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Завантажити ваги моделі у checkpoints/sam/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Завантажити всі моделі (mobile_sam, sam_b, sam_l)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SAM_MODEL,
        help="mobile_sam.pt, sam_b.pt або sam_l.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    model = args.model
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.is_file():
            from src.utils.config import load_config

            cfg = load_config(cfg_path)
            model = resolve_sam_model_name(cfg.get("roi", {}))

    if args.download:
        if args.all:
            for name in SAM_MODEL_CHOICES:
                prepare_sam_model(name, verbose=True)
        else:
            prepare_sam_model(model, verbose=True)
        return

    err = ultralytics_import_error()
    if err:
        print(err)
        raise SystemExit(1)
    print(f"ultralytics OK, каталог SAM: {SAM_DIR}")
    print(f"Модель за замовчуванням: {model}")
    print("Завантаження ваг: make setup-sam")


if __name__ == "__main__":
    _main()
