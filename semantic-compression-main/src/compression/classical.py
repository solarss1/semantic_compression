"""Класичні методи стиснення з вимірюванням часу кодування/декодування."""

import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.evaluation.metrics import compression_ratio


@dataclass
class CompressionResult:
    """Результат одного методу стиснення."""

    name: str
    reconstructed: np.ndarray
    bitstream: bytes
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    extra: dict | None = None

    @property
    def compression_ratio(self) -> float:
        orig = (self.extra or {}).get("original_bytes", 1)
        return compression_ratio(orig, self.compressed_bytes)


def _original_bytes(image: np.ndarray) -> int:
    from src.evaluation.metrics import raw_image_bytes

    return raw_image_bytes(image)


def _encode_decode(
    name: str,
    image: np.ndarray,
    encode_fn,
    decode_fn,
) -> CompressionResult:
    orig_bytes = _original_bytes(image)
    t0 = time.perf_counter()
    bitstream = encode_fn(image)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    reconstructed = decode_fn(bitstream)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    return CompressionResult(
        name=name,
        reconstructed=reconstructed,
        bitstream=bitstream,
        compressed_bytes=len(bitstream),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        extra={"original_bytes": orig_bytes},
    )


def compress_jpeg(image: np.ndarray, quality: int = 50) -> CompressionResult:
    def encode(img: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(
            buf, format="JPEG", quality=quality, optimize=True
        )
        return buf.getvalue()

    def decode(data: bytes) -> np.ndarray:
        buf = io.BytesIO(data)
        return np.asarray(Image.open(buf).convert("RGB"))

    return _encode_decode(f"jpeg_q{quality}", image, encode, decode)


def compress_webp(image: np.ndarray, quality: int = 50) -> CompressionResult:
    def encode(img: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(
            buf, format="WEBP", quality=quality, method=4
        )
        return buf.getvalue()

    def decode(data: bytes) -> np.ndarray:
        buf = io.BytesIO(data)
        return np.asarray(Image.open(buf).convert("RGB"))

    return _encode_decode(f"webp_q{quality}", image, encode, decode)


def compress_png(image: np.ndarray, compress_level: int = 6) -> CompressionResult:
    def encode(img: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(
            buf, format="PNG", compress_level=compress_level
        )
        return buf.getvalue()

    def decode(data: bytes) -> np.ndarray:
        buf = io.BytesIO(data)
        return np.asarray(Image.open(buf).convert("RGB"))

    return _encode_decode(f"png_l{compress_level}", image, encode, decode)
