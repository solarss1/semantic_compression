import numpy as np

from src.roi.mask_utils import smooth_roi_mask


def test_smooth_roi_mask_preserves_range():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:44, 20:44] = 1.0
    out = smooth_roi_mask(mask, sigma=3.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.dtype == np.float32


def test_smooth_roi_sigma_zero():
    mask = np.ones((32, 32), dtype=np.float32) * 0.5
    out = smooth_roi_mask(mask, sigma=0)
    np.testing.assert_array_equal(out, mask)
