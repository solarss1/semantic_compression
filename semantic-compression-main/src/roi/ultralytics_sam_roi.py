"""ROI через Ultralytics SAM (point / box prompts)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.roi.base import ROIExtractor
from src.roi.ultralytics_sam_setup import (
    UltralyticsSamNotAvailableError,
    format_sam_point_prompts,
    get_sam_model,
    masks_to_float32,
    parse_sam_boxes,
    parse_sam_points,
    resolve_sam_model_name,
    ultralytics_import_error,
)


class UltralyticsSamROIExtractor(ROIExtractor):
    def __init__(
        self,
        *,
        device: str = "cpu",
        model_name: str = "mobile_sam.pt",
        points: list[tuple[int, int, int]] | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.points = list(points or [])
        self.boxes = list(boxes or [])

    @classmethod
    def from_config(cls, roi_cfg: dict[str, Any]) -> UltralyticsSamROIExtractor:
        device = str(roi_cfg.get("device", "cpu"))
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            device=device,
            model_name=resolve_sam_model_name(roi_cfg),
            points=parse_sam_points(roi_cfg.get("sam_points")),
            boxes=parse_sam_boxes(roi_cfg.get("sam_boxes")),
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        if not self.points and not self.boxes:
            raise ValueError(
                "Ultralytics SAM: додайте клік(и) або намалюйте bbox на зображенні "
                "(режим «Оригінал»)."
            )
        err = ultralytics_import_error()
        if err:
            raise UltralyticsSamNotAvailableError(err)

        h, w = image.shape[:2]
        model = get_sam_model(self.model_name, self.device)

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "verbose": False,
            "device": self.device,
        }
        if self.points:
            pt_coords, pt_labels = format_sam_point_prompts(self.points)
            predict_kwargs["points"] = pt_coords
            predict_kwargs["labels"] = pt_labels
        if self.boxes:
            predict_kwargs["bboxes"] = [list(box) for box in self.boxes]

        results = model.predict(**predict_kwargs)
        if not results:
            raise RuntimeError("Ultralytics SAM не повернув результат.")

        merged = np.zeros((h, w), dtype=np.float32)
        for result in results:
            if result.masks is None:
                continue
            data = result.masks.data
            if data is None:
                continue
            merged = np.maximum(merged, masks_to_float32(data, h, w))

        if merged.max() <= 0:
            raise RuntimeError(
                "Ultralytics SAM не знайшов об'єкт за промптами. "
                "Спробуйте інший клік або bbox."
            )
        return merged
