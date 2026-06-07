import numpy as np

from src.compression.classical import compress_jpeg
from src.compression.jpeg_utils import jpeg_encode
from src.compression.semantic import SemanticCompressor


def test_semantic_file_size_is_single_jpeg():
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:256, :256]
    mask = ((yy - 128) ** 2 + (xx - 128) ** 2 < 40**2).astype(np.float32)

    compressor = SemanticCompressor(quality_roi=85, quality_background=35, tile_size=32)
    _, bitstream, _ = compressor.compress(img, mask)

    tile_sum = 0
    tile = 32
    for y in range(0, 256, tile):
        for x in range(0, 256, tile):
            patch = img[y : y + tile, x : x + tile]
            w = float(mask[y : y + tile, x : x + tile].mean())
            q = int(35 + w * (85 - 35))
            tile_sum += len(jpeg_encode(patch, max(1, min(100, q))))

    assert len(bitstream) < tile_sum * 0.6
    assert len(bitstream) > 500


def test_semantic_ratio_in_same_order_as_classical():
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[200:312, 200:312] = 1.0

    classical = compress_jpeg(img, quality=50)
    _, sem_bytes, _ = SemanticCompressor(85, 35, 64).compress(img, mask)

    classical_ratio = classical.compression_ratio
    semantic_ratio = img.size / len(sem_bytes)

    assert semantic_ratio > classical_ratio * 0.25
    assert semantic_ratio < classical_ratio * 3.0
