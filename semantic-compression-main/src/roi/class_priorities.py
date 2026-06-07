"""Пріоритети семантичних класів COCO для зваженої ROI-маски."""

from __future__ import annotations

from typing import Any

# За замовчуванням: люди та тварини — найвищий пріоритет
DEFAULT_CLASS_WEIGHTS: dict[int, float] = {
    1: 1.00,  # person
    17: 0.95,
    18: 0.95,
    19: 0.90,
    16: 0.90,  # animals
    3: 0.80,
    4: 0.80,
    2: 0.75,
    6: 0.75,
    8: 0.75,  # vehicles
    64: 0.55,
    62: 0.50,
    63: 0.50,
    65: 0.45,
    66: 0.45,
    67: 0.45,  # furniture / interior
}

DEFAULT_PRIORITY_TIERS: dict[str, list[int]] = {
    "high": [1],
    "medium": [2, 3, 4, 6, 8, 16, 17, 18, 19],
    "low": [62, 63, 64, 65, 66, 67],
}

DEFAULT_PRIORITY_VALUES: dict[str, float] = {
    "high": 1.0,
    "medium": 0.75,
    "low": 0.45,
}


def parse_class_weights(roi_cfg: dict[str, Any]) -> dict[int, float]:
    """
    Зібрати ваги класів з config:
    - class_weights: {1: 1.0, 3: 0.8, ...}
    - або class_priority + priority_values (рівні пріоритету)
    """
    raw = roi_cfg.get("class_weights")
    if raw:
        return {int(k): float(v) for k, v in raw.items()}

    tiers = roi_cfg.get("class_priority") or DEFAULT_PRIORITY_TIERS
    values = roi_cfg.get("priority_values") or DEFAULT_PRIORITY_VALUES

    weights: dict[int, float] = {}
    for tier_name, class_ids in tiers.items():
        w = float(values.get(tier_name, 0.5))
        for cls_id in class_ids:
            weights[int(cls_id)] = w

    return weights or dict(DEFAULT_CLASS_WEIGHTS)


def target_class_ids(weights: dict[int, float]) -> set[int]:
    return set(weights.keys())
