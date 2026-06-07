"""Тести збереження зображень."""

import numpy as np

from src.compression.classical import compress_jpeg, compress_png, compress_webp
from src.utils.io import resolve_image_format, save_compression_result, save_image


def test_save_image_adds_extension(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    path = save_image(img, tmp_path / "out", default_ext="png")
    assert path.suffix == ".png"
    assert path.exists()


def test_save_image_jpeg_explicit(tmp_path):
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    path = save_image(img, tmp_path / "photo.jpg", format="JPEG")
    assert path.exists()
    assert path.stat().st_size > 0


def test_resolve_image_format_no_suffix():
    fmt, path = resolve_image_format(__import__("pathlib").Path("file"), "jpg")
    assert fmt == "JPEG"
    assert path.suffix == ".jpg"


def test_save_compression_preserves_jpeg_bitstream(tmp_path):
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    result = compress_jpeg(img, quality=55)
    path = save_compression_result(
        tmp_path / "out.jpg",
        bitstream=result.bitstream,
        reconstructed=result.reconstructed,
        source_extension="jpg",
        target_extension="jpg",
    )
    assert path.read_bytes() == result.bitstream
    assert path.stat().st_size == result.compressed_bytes


def test_save_compression_preserves_webp_bitstream(tmp_path):
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    result = compress_webp(img, quality=40)
    path = save_compression_result(
        tmp_path / "out.webp",
        bitstream=result.bitstream,
        reconstructed=result.reconstructed,
        source_extension="webp",
        target_extension="webp",
    )
    assert path.read_bytes() == result.bitstream


def test_save_compression_preserves_png_bitstream(tmp_path):
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    result = compress_png(img, compress_level=6)
    path = save_compression_result(
        tmp_path / "out.png",
        bitstream=result.bitstream,
        reconstructed=result.reconstructed,
        source_extension="png",
        target_extension="png",
    )
    assert path.read_bytes() == result.bitstream


def test_save_compression_reencodes_on_format_change(tmp_path):
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    result = compress_jpeg(img, quality=55)
    path = save_compression_result(
        tmp_path / "out.png",
        bitstream=result.bitstream,
        reconstructed=result.reconstructed,
        source_extension="jpg",
        target_extension="png",
    )
    saved = path.read_bytes()
    assert saved != result.bitstream
    assert saved[:8] == b"\x89PNG\r\n\x1a\n"
