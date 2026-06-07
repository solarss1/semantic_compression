import numpy as np

from src.compression.classical import compress_jpeg


def test_compress_jpeg_roundtrip():
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    r = compress_jpeg(img, quality=75)
    assert r.compressed_bytes > 0
    assert r.reconstructed.shape == img.shape
    assert r.compression_ratio > 1.0
    assert r.encode_ms >= 0
