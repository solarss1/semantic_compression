import numpy as np
import pytest

from src.app.models import AppSettings
from src.app.service import compress_classical, compress_semantic
from src.compression.classical import compress_jpeg
from src.evaluation.metrics import (
    bits_per_pixel,
    compression_ratio,
    evaluate_pair,
    raw_image_bytes,
)


def test_raw_image_bytes_rgb():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert raw_image_bytes(img) == 100 * 200 * 3


def test_raw_image_bytes_rejects_non_rgb():
    with pytest.raises(ValueError):
        raw_image_bytes(np.zeros((10, 10), dtype=np.uint8))


def test_bits_per_pixel_formula():
    assert bits_per_pixel(1250, 100, 100) == pytest.approx(1.0)


def test_compression_ratio_formula():
    assert compression_ratio(3000, 1000) == pytest.approx(3.0)
    assert compression_ratio(500, 0) == pytest.approx(500.0)


def test_evaluate_pair_identical():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    metrics = evaluate_pair(img, img.copy(), file_size_bytes=1000)
    assert metrics.psnr == float("inf") or metrics.psnr > 50
    assert metrics.ssim > 0.99
    assert metrics.bpp == bits_per_pixel(1000, 64, 64)
    assert metrics.file_size_bytes == 1000


def test_evaluate_pair_shape_mismatch():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        evaluate_pair(a, b, file_size_bytes=100)


def test_evaluate_pair_with_roi_mask():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :] = 255
    noisy = img.copy()
    noisy[16:, :] = 128
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[:16, :] = 1.0
    metrics = evaluate_pair(img, noisy, file_size_bytes=500, roi_mask=mask)
    assert metrics.roi_psnr == float("inf") or metrics.roi_psnr > 40
    assert metrics.background_psnr is not None


def test_classical_output_metrics_are_consistent():
    img = np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8)
    out = compress_classical(img, AppSettings(classical_method="jpeg", classical_quality=55))
    orig = raw_image_bytes(img)
    compressed = out.metrics.file_size_bytes

    assert out.original_bytes == orig
    assert len(out.bitstream) == compressed
    assert out.metrics.bpp == bits_per_pixel(compressed, 128, 192)
    assert out.compression_ratio == pytest.approx(orig / compressed)
    assert out.compression_ratio * compressed == pytest.approx(orig)


def test_semantic_output_metrics_are_consistent():
    img = np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8)
    mask = np.zeros((128, 192), dtype=np.float32)
    mask[40:88, 60:132] = 1.0
    out = compress_semantic(img, mask, AppSettings())
    orig = raw_image_bytes(img)
    compressed = out.metrics.file_size_bytes

    assert out.original_bytes == orig
    assert len(out.bitstream) == compressed
    assert out.compression_ratio == pytest.approx(orig / compressed)


def test_jpeg_result_matches_evaluate_pair():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = compress_jpeg(img, quality=60)
    metrics = evaluate_pair(img, result.reconstructed, result.compressed_bytes)
    assert metrics.file_size_bytes == len(result.bitstream)
    assert result.compression_ratio == pytest.approx(
        raw_image_bytes(img) / result.compressed_bytes
    )
