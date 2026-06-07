"""Тести Ultralytics SAM (парсери, без GPU)."""

import numpy as np
import pytest

from src.roi.ultralytics_sam_roi import UltralyticsSamROIExtractor
from src.roi.ultralytics_sam_setup import (
    SAM_DIR,
    format_sam_point_prompts,
    masks_to_float32,
    parse_sam_boxes,
    parse_sam_points,
    sam_model_path,
    scale_sam_prompts,
    ultralytics_import_error,
)


def test_parse_sam_points():
    raw = [[10, 20, 1], [5, 6, 0], [1, 2]]
    assert parse_sam_points(raw) == [(10, 20, 1), (5, 6, 0)]
    assert parse_sam_points(None) == []


def test_parse_sam_boxes_normalizes_order():
    assert parse_sam_boxes([[100, 50, 10, 20]]) == [(10, 20, 100, 50)]
    assert parse_sam_boxes([[0, 0, 0, 0]]) == []


def test_masks_to_float32_merge():
    m1 = np.zeros((4, 4), dtype=np.float32)
    m1[0, 0] = 1.0
    m2 = np.zeros((4, 4), dtype=np.float32)
    m2[1, 1] = 0.5
    merged = masks_to_float32(np.stack([m1, m2]), 4, 4)
    assert merged[0, 0] == 1.0
    assert merged[1, 1] == 0.5


def test_format_sam_point_prompts_nested():
    pts = [(10, 20, 1), (30, 40, 0)]
    coords, labels = format_sam_point_prompts(pts)
    assert coords == [[[10, 20], [30, 40]]]
    assert labels == [[1, 0]]


def test_scale_sam_prompts():
    points = [(100, 200, 1)]
    boxes = [(0, 0, 100, 100)]
    sp, sb = scale_sam_prompts((1000, 2000), (500, 1000), points, boxes)
    assert sp == [(50, 100, 1)]
    assert sb == [(0, 0, 50, 50)]


def test_sam_model_path_in_checkpoints():
    path = sam_model_path("mobile_sam.pt")
    assert path.name == "mobile_sam.pt"
    assert path.parent == SAM_DIR
    assert "checkpoints" in path.parts
    assert path.parent.name == "sam"


def test_sam_extractor_requires_prompts():
    ext = UltralyticsSamROIExtractor(device="cpu", points=[], boxes=[])
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Ultralytics SAM"):
        ext.extract(image)


@pytest.mark.skipif(ultralytics_import_error() is None, reason="ultralytics installed")
def test_ultralytics_import_error_when_missing():
    assert ultralytics_import_error() is not None
