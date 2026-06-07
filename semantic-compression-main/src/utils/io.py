from pathlib import Path

import numpy as np
from PIL import Image

_PIL_FORMATS: dict[str, str] = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tif": "TIFF",
    "tiff": "TIFF",
}


def load_image(path: str | Path) -> np.ndarray:
    """Завантажити RGB зображення як ndarray uint8 (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def extension_for_format(pil_format: str) -> str:
    """JPEG → jpg, PNG → png, …"""
    mapping = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tif"}
    return mapping.get(pil_format.upper(), pil_format.lower())


def resolve_image_format(path: Path, default_ext: str = "png") -> tuple[str, Path]:
    """
    Визначити формат Pillow і доповнити шлях розширенням, якщо його немає.
    """
    ext = path.suffix.lower().lstrip(".")
    if ext in _PIL_FORMATS:
        return _PIL_FORMATS[ext], path
    ext = default_ext.lower().lstrip(".")
    return _PIL_FORMATS.get(ext, "PNG"), path.with_suffix(f".{ext}")


def save_image(
    image: np.ndarray,
    path: str | Path,
    *,
    format: str | None = None,
    default_ext: str = "png",
    jpeg_quality: int = 90,
) -> Path:
    """Зберегти RGB ndarray у файл. Повертає фактичний шлях з розширенням."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        pil_format, path = resolve_image_format(path, default_ext)
    else:
        pil_format = format.upper()
        if not path.suffix:
            path = path.with_suffix(f".{extension_for_format(pil_format)}")

    arr = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)

    save_kwargs: dict = {}
    if pil_format == "JPEG":
        save_kwargs["quality"] = jpeg_quality
        save_kwargs["optimize"] = True
    elif pil_format == "PNG":
        save_kwargs["compress_level"] = 6
    elif pil_format == "WEBP":
        save_kwargs["quality"] = jpeg_quality

    pil.save(path, format=pil_format, **save_kwargs)
    return path


def extensions_match(path_ext: str, source_ext: str) -> bool:
    """Чи збігається розширення файлу з форматом результату стиснення."""
    a = path_ext.lower().lstrip(".")
    b = source_ext.lower().lstrip(".")
    if a == b:
        return True
    return {a, b} <= {"jpg", "jpeg"}


def save_compression_result(
    path: str | Path,
    *,
    bitstream: bytes | None,
    reconstructed: np.ndarray,
    source_extension: str,
    target_extension: str,
    jpeg_quality: int = 90,
) -> Path:
    """
    Зберегти результат стиснення.

    Якщо цільовий формат збігається з результатом кодування — записує bitstream
    без повторного стиснення. Інакше — перекодування у вибраний формат.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(f".{target_extension}")

    path_ext = path.suffix.lower().lstrip(".")
    if bitstream and extensions_match(path_ext, source_extension):
        return write_bytes(path, bitstream, source_extension)

    pil_format, path = resolve_image_format(path, default_ext=target_extension)
    return save_image(
        reconstructed,
        path,
        format=pil_format,
        default_ext=target_extension,
        jpeg_quality=jpeg_quality,
    )


def write_bytes(path: str | Path, data: bytes, default_ext: str = "bin") -> Path:
    """Записати готовий bitstream; додати розширення, якщо відсутнє."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(f".{default_ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
