from abc import ABC, abstractmethod

import numpy as np


class ROIExtractor(ABC):
    """Базовий інтерфейс для виділення областей інтересу (ROI)."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Повертає маску ROI зі значеннями у діапазоні [0, 1].

        Args:
            image: RGB зображення (H, W, 3), uint8.

        Returns:
            Маска (H, W), float32 у [0, 1].
        """
        ...
