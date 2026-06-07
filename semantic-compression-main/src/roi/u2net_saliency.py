"""Салієнтність на основі U²-Net (локальне завантаження, без torch.hub cache)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.roi.base import ROIExtractor
from src.roi.saliency import SaliencyROIExtractor
from src.roi.u2net_setup import load_u2net_model

_U2NET_MODEL = None
_U2NET_DEVICE: str | None = None
_U2NET_FAILED = False

U2NET_INPUT_SIZE = 320


def _load_u2net(device: str):
    global _U2NET_MODEL, _U2NET_DEVICE, _U2NET_FAILED
    if _U2NET_FAILED:
        raise RuntimeError("U²-Net previously failed to load")
    if _U2NET_MODEL is not None and _U2NET_DEVICE == device:
        return _U2NET_MODEL
    _U2NET_MODEL = load_u2net_model(device, verbose=False)
    _U2NET_DEVICE = device
    return _U2NET_MODEL


class U2NetSaliencyROIExtractor(ROIExtractor):
    """Карта салієнтності з U²-Net."""

    def __init__(self, device: str | None = None, input_size: int = U2NET_INPUT_SIZE) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self._model = None
        self._normalize = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _get_model(self):
        if self._model is None:
            self._model = _load_u2net(self.device)
        return self._model

    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pil = Image.fromarray(image.astype(np.uint8))
        tensor = self._normalize(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self._get_model()(tensor)
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            sal = pred[:, 0:1, :, :]
            sal = F.interpolate(
                sal, size=(h, w), mode="bilinear", align_corners=False
            )
            sal = sal.squeeze().cpu().numpy().astype(np.float32)

        sal = np.clip(sal, 0.0, 1.0)
        if sal.max() > sal.min():
            sal = (sal - sal.min()) / (sal.max() - sal.min())
        return sal


def create_saliency_extractor(
    model_name: str,
    device: str | None = None,
) -> ROIExtractor:
    """u2net (локально) або resnet (fallback)."""
    global _U2NET_FAILED
    name = (model_name or "u2net").lower()
    if name in ("u2net", "u2-net", "u_square_net"):
        if _U2NET_FAILED:
            return SaliencyROIExtractor(device=device)
        try:
            return U2NetSaliencyROIExtractor(device=device)
        except Exception as exc:
            _U2NET_FAILED = True
            print(f"  U²-Net недоступний ({exc}), fallback → ResNet saliency")
            return SaliencyROIExtractor(device=device)
    if name in ("resnet", "resnet50", "saliency_resnet"):
        return SaliencyROIExtractor(device=device)
    raise ValueError(f"Unknown saliency model: {model_name}")
