"""Виділення ROI за допомогою карти салієнтності (глибока модель)."""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from src.roi.base import ROIExtractor


class SaliencyROIExtractor(ROIExtractor):
    """
    ROI на основі семантично значущих ознак з попередньо навченої CNN.

    Використовує градієнти активацій останнього згорткового шару
    (class-agnostic saliency) як наближення карти уваги.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocess = weights.transforms()
        self.target_layer = self.model.layer4[-1]

    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pil_image = transforms.functional.to_pil_image(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        activations: torch.Tensor | None = None

        def hook(_module, _input, output: torch.Tensor) -> None:
            nonlocal activations
            activations = output

        handle = self.target_layer.register_forward_hook(hook)
        try:
            self.model(tensor)
            if activations is None:
                raise RuntimeError("Failed to capture activations")

            score = activations.pow(2).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()

            saliency = tensor.grad.abs().sum(dim=1).squeeze(0)
            saliency = F.interpolate(
                saliency.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            saliency = saliency.cpu().numpy().astype(np.float32)
            saliency = (saliency - saliency.min()) / (
                saliency.max() - saliency.min() + 1e-8
            )
            result = saliency
            return result
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
            if tensor.grad is not None:
                tensor.grad = None
