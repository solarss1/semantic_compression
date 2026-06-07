"""Розмір «оригіналу»: файл на диску vs нестиснений RGB."""

import numpy as np
import pytest
from PIL import Image

from src.evaluation.metrics import raw_image_bytes


def test_raw_rgb_larger_than_typical_jpeg_on_disk(tmp_path):
    """768×512 RGB ≈ 1152 KB у пам'яті; JPEG на диску зазвичай менший."""
    img = np.random.randint(0, 256, (512, 768, 3), dtype=np.uint8)
    jpeg_path = tmp_path / "sample.jpg"
    Image.fromarray(img).save(jpeg_path, format="JPEG", quality=85)

    disk_bytes = jpeg_path.stat().st_size
    rgb_bytes = raw_image_bytes(img)

    assert rgb_bytes == 512 * 768 * 3
    assert rgb_bytes / 1024 == pytest.approx(1152.0)
    assert disk_bytes < rgb_bytes


def test_kodak_sample_disk_vs_rgb():
    from pathlib import Path

    samples = list(Path("data/samples").glob("*.png"))
    if not samples:
        pytest.skip("no samples")
    path = samples[0]
    img = np.asarray(Image.open(path).convert("RGB"))
    disk = path.stat().st_size
    rgb = raw_image_bytes(img)
    assert rgb >= disk * 0.5
