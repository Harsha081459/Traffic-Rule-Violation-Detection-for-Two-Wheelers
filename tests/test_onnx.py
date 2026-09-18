import numpy as np
import pytest

from traffic_violation.models.onnx_detector import ONNXDetector

pytestmark = pytest.mark.unit


@pytest.fixture
def detector():
    model = ONNXDetector.__new__(ONNXDetector)
    model._names = {0: "helmet", 1: "no_helmet"}
    model._transposed = True
    return model


@pytest.mark.parametrize("height,width", [(1, 2000), (2000, 1), (333, 517)])
def test_letterbox_handles_extreme_and_non_stride_aspect_ratios(detector, height, width):
    blob, info = detector._preprocess(np.zeros((height, width, 3), dtype=np.uint8), 640)
    assert blob.shape == (1, 3, 640, 640)
    scale, _, _, resized_h, resized_w = info
    assert abs(resized_h - height * scale) <= 1
    assert abs(resized_w - width * scale) <= 1


def test_nms_keeps_overlapping_different_classes(detector):
    raw = np.array([[[100, 100, 30, 30, 0.9, 0.1], [100, 100, 30, 30, 0.1, 0.8]]], dtype=np.float32)
    result = detector._postprocess(raw, 0.25, 0.5, 640, 640, (1, 0, 0, 640, 640))
    assert {item.cls_id for item in result} == {0, 1}


def test_nms_removes_duplicates_of_same_class(detector):
    raw = np.array([[[100, 100, 30, 30, 0.9, 0.1], [100, 100, 30, 30, 0.8, 0.1]]], dtype=np.float32)
    assert len(detector._postprocess(raw, 0.25, 0.5, 640, 640, (1, 0, 0, 640, 640))) == 1


def test_transposed_layouts_give_same_boxes(detector):
    raw = np.array([[[100, 100, 30, 30, 0.9, 0.1]]], dtype=np.float32)
    expected = detector._postprocess(raw, 0.25, 0.5, 640, 640, (1, 0, 0, 640, 640))
    detector._transposed = False
    actual = detector._postprocess(raw.transpose(0, 2, 1), 0.25, 0.5, 640, 640, (1, 0, 0, 640, 640))
    assert actual[0].xyxy == expected[0].xyxy
