"""Обробка масок ROI."""

import numpy as np


def smooth_roi_mask(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Згладжування маски Gaussian blur для м'якших переходів якості JPEG.

    Args:
        mask: float32 [0, 1], shape (H, W)
        sigma: стандартне відхилення (0 = без змін)
    """
    if sigma <= 0:
        return mask.astype(np.float32)

    import cv2

    k = int(6 * sigma + 1) | 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigma)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)
