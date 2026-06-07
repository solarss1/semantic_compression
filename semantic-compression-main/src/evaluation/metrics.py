from dataclasses import dataclass

import numpy as np


@dataclass
class CompressionMetrics:
    psnr: float
    ssim: float
    bpp: float
    file_size_bytes: int
    roi_psnr: float | None = None
    background_psnr: float | None = None


def raw_image_bytes(image: np.ndarray) -> int:
    """Нестиснений RGB uint8: H × W × 3 байти."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Очікується RGB (H, W, 3), отримано {image.shape}")
    return int(image.shape[0] * image.shape[1] * image.shape[2])


def bits_per_pixel(file_size_bytes: int, height: int, width: int) -> float:
    return (file_size_bytes * 8) / (height * width)


def compression_ratio(original_bytes: int, file_size_bytes: int) -> float:
    return original_bytes / max(file_size_bytes, 1)


def _psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff * diff)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255**2 / mse))


def _ssim_channel(a: np.ndarray, b: np.ndarray, window: int = 11) -> float:
    """SSIM для одного каналу (numpy, без scipy)."""
    from numpy.lib.stride_tricks import sliding_window_view

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    pad = window // 2
    a_pad = np.pad(a, pad, mode="reflect")
    b_pad = np.pad(b, pad, mode="reflect")

    pa = sliding_window_view(a_pad, (window, window))
    pb = sliding_window_view(b_pad, (window, window))

    mu_a = pa.mean(axis=(-2, -1))
    mu_b = pb.mean(axis=(-2, -1))
    sigma_a = ((pa - mu_a[..., None, None]) ** 2).mean(axis=(-2, -1))
    sigma_b = ((pb - mu_b[..., None, None]) ** 2).mean(axis=(-2, -1))
    sigma_ab = (
        (pa - mu_a[..., None, None]) * (pb - mu_b[..., None, None])
    ).mean(axis=(-2, -1))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(num / den))


def _ssim_rgb(original: np.ndarray, reconstructed: np.ndarray) -> float:
    scores = [
        _ssim_channel(original[..., c], reconstructed[..., c])
        for c in range(original.shape[-1])
    ]
    return float(np.mean(scores))


def evaluate_pair(
    original: np.ndarray,
    reconstructed: np.ndarray,
    file_size_bytes: int,
    roi_mask: np.ndarray | None = None,
) -> CompressionMetrics:
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Розмір original {original.shape} != reconstructed {reconstructed.shape}"
        )
    h, w = original.shape[:2]
    pixels = h * w

    psnr = _psnr(original, reconstructed)
    ssim = _ssim_rgb(original, reconstructed)
    bpp = bits_per_pixel(file_size_bytes, h, w)

    roi_psnr = None
    bg_psnr = None
    if roi_mask is not None:
        binary = roi_mask >= 0.5
        if binary.any():
            roi_psnr = _masked_psnr(original, reconstructed, binary)
        bg_mask = ~binary
        if bg_mask.any():
            bg_psnr = _masked_psnr(original, reconstructed, bg_mask)

    return CompressionMetrics(
        psnr=psnr,
        ssim=ssim,
        bpp=bpp,
        file_size_bytes=file_size_bytes,
        roi_psnr=roi_psnr,
        background_psnr=bg_psnr,
    )


def _masked_psnr(
    original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray
) -> float:
    mask = mask.astype(bool)
    diff = (original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    mse = diff[mask].mean()
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255**2 / mse))
