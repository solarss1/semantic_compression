"""Тести допоміжних функцій desktop-додатку."""

import numpy as np

from src.app.service import compression_diff, mask_to_rgb, roi_overlay, sam_prompts_overlay


def test_mask_to_rgb():
    mask = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    rgb = mask_to_rgb(mask)
    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 1, 0] == 255


def test_compression_diff():
    orig = np.zeros((4, 4, 3), dtype=np.uint8)
    recon = orig.copy()
    recon[0, 0] = 10
    diff = compression_diff(orig, recon, amplify=10.0)
    assert diff[0, 0, 0] == 100


def test_roi_overlay():
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    out = roi_overlay(img, mask)
    assert out.shape == img.shape


def test_sam_prompts_overlay():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = sam_prompts_overlay(
        img,
        points=[(8, 8, 1), (20, 20, 0)],
        boxes=[(10, 10, 18, 18)],
        draft_box=(2, 2, 8, 8),
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)
    assert out[8, 8].any() > 0
    assert out[10, 14].any() > 0
    assert out[2, 4].any() > 0
