"""Виділення ROI за семантичною сегментацією (DeepLabV3) з пріоритетами класів."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from src.roi.base import ROIExtractor
from src.roi.class_priorities import (
    DEFAULT_CLASS_WEIGHTS,
    parse_class_weights,
    target_class_ids,
)


class SegmentationROIExtractor(ROIExtractor):
    """
    ROI з DeepLabV3 (COCO).

    Режими:
    - бінарна маска (use_class_weights=False): клас з whitelist = 1
    - зважена (use_class_weights=True): вага класу задає «важливість» пікселя [0,1]
    """

    def __init__(
        self,
        model_name: str = "deeplabv3_resnet50",
        target_classes: set[int] | None = None,
        class_weights: dict[int, float] | None = None,
        use_class_weights: bool = True,
        device: str | None = None,
        softmax_blend: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_class_weights = use_class_weights
        self.softmax_blend = softmax_blend

        if class_weights is not None:
            self.class_weights = dict(class_weights)
        elif target_classes is not None:
            self.class_weights = {c: 1.0 for c in target_classes}
        else:
            self.class_weights = dict(DEFAULT_CLASS_WEIGHTS)

        self.target_classes = target_classes or target_class_ids(self.class_weights)

        if model_name != "deeplabv3_resnet50":
            raise ValueError(f"Unsupported model: {model_name}")

        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights).to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self._weight_tensor = self._build_weight_tensor()

    def _build_weight_tensor(self) -> torch.Tensor:
        """Тензор ваг (21 клас COCO + background)."""
        n_classes = 21
        w = torch.zeros(n_classes, dtype=torch.float32)
        for cls_id, weight in self.class_weights.items():
            if 0 <= cls_id < n_classes:
                w[cls_id] = float(weight)
        return w

    def _fallback_center_mask(self, h: int, w: int) -> np.ndarray:
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        return (dist < min(h, w) * 0.25).astype(np.float32)

    def _mask_from_labels(self, labels: np.ndarray) -> np.ndarray:
        h, w = labels.shape
        mask = np.zeros((h, w), dtype=np.float32)
        if self.use_class_weights:
            for cls_id, weight in self.class_weights.items():
                mask[labels == cls_id] = np.maximum(
                    mask[labels == cls_id], weight
                )
        else:
            for cls_id in self.target_classes:
                mask[labels == cls_id] = 1.0
        return mask

    def _mask_from_logits(self, logits: torch.Tensor) -> np.ndarray:
        """Зважена маска через softmax (м'якші межі класів)."""
        probs = F.softmax(logits, dim=1)  # 1, C, H, W
        w = self._weight_tensor.to(logits.device).view(1, -1, 1, 1)
        # Сума ймовірностей зважених класів, обмежена [0, 1]
        weighted = (probs * w).sum(dim=1, keepdim=True)
        max_w = self._weight_tensor.max().clamp(min=1e-6)
        mask = (weighted / max_w).squeeze().cpu().numpy().astype(np.float32)
        return np.clip(mask, 0.0, 1.0)

    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pil_image = transforms.functional.to_pil_image(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)["out"]
            output = F.interpolate(
                output, size=(h, w), mode="bilinear", align_corners=False
            )

            if self.use_class_weights and self.softmax_blend:
                mask = self._mask_from_logits(output)
            else:
                labels = output.argmax(dim=1).squeeze(0).cpu().numpy()
                mask = self._mask_from_labels(labels)

        if mask.max() < 1e-6:
            mask = self._fallback_center_mask(h, w)

        return mask.astype(np.float32)


def segmentation_extractor_from_config(
    roi_cfg: dict,
    device: str | None,
    *,
    weighted: bool | None = None,
) -> SegmentationROIExtractor:
    """Побудувати екстрактор з блоку roi config."""
    use_weights = weighted if weighted is not None else roi_cfg.get(
        "use_class_weights", True
    )
    class_weights = parse_class_weights(roi_cfg)
    target = roi_cfg.get("target_classes") or []
    if target and not roi_cfg.get("class_weights") and not roi_cfg.get("class_priority"):
        class_weights = {int(c): 1.0 for c in target}

    return SegmentationROIExtractor(
        model_name=roi_cfg.get("segmentation_model", "deeplabv3_resnet50"),
        class_weights=class_weights,
        use_class_weights=use_weights,
        device=device,
        softmax_blend=bool(roi_cfg.get("segmentation_softmax_blend", True)),
    )
