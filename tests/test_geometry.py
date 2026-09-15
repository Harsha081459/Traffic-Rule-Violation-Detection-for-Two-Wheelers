"""Unit tests for traffic_violation.utils.geometry — pure functions, no I/O."""

import numpy as np
import pytest

from traffic_violation.utils.geometry import (
    Det,
    clip_box,
    crop_box,
    expand_box,
    inter_area,
    iou,
    nms_same_class,
    norm_name,
    offset_det,
)


class TestClipBox:
    @pytest.mark.unit
    def test_clips_to_image_bounds(self):
        assert clip_box([-5, -10, 120, 250], w=100, h=200) == [0, 0, 99, 199]

    @pytest.mark.unit
    def test_box_inside_bounds_unchanged(self):
        assert clip_box([10.2, 20.4, 50.6, 80.9], w=100, h=200) == [10, 20, 51, 81]

    @pytest.mark.unit
    def test_degenerate_box_gets_min_size(self):
        out = clip_box([50, 50, 50, 50], w=100, h=100)
        assert out[2] > out[0]
        assert out[3] > out[1]


class TestInterArea:
    @pytest.mark.unit
    def test_overlap(self):
        assert inter_area([0, 0, 10, 10], [5, 5, 15, 15]) == 25.0

    @pytest.mark.unit
    def test_no_overlap(self):
        assert inter_area([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    @pytest.mark.unit
    def test_contained(self):
        assert inter_area([0, 0, 10, 10], [2, 2, 8, 8]) == 36.0


class TestIoU:
    @pytest.mark.unit
    def test_identical_boxes(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_half_overlap(self):
        # inter=50, union=150
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50.0 / 150.0)

    @pytest.mark.unit
    def test_disjoint(self):
        assert iou([0, 0, 1, 1], [5, 5, 9, 9]) == 0.0


class TestNmsSameClass:
    @pytest.mark.unit
    def test_suppresses_lower_conf_same_class(self):
        dets = [
            Det(cls_id=0, cls_name="helmet", conf=0.9, xyxy=[0, 0, 10, 10]),
            Det(cls_id=0, cls_name="helmet", conf=0.5, xyxy=[1, 1, 11, 11]),
        ]
        kept = nms_same_class(dets, iou_thr=0.45)
        assert len(kept) == 1
        assert kept[0].conf == 0.9

    @pytest.mark.unit
    def test_keeps_overlapping_different_class(self):
        dets = [
            Det(cls_id=0, cls_name="helmet", conf=0.9, xyxy=[0, 0, 10, 10]),
            Det(cls_id=1, cls_name="rider", conf=0.5, xyxy=[1, 1, 11, 11]),
        ]
        assert len(nms_same_class(dets, iou_thr=0.45)) == 2

    @pytest.mark.unit
    def test_keeps_disjoint_same_class(self):
        dets = [
            Det(cls_id=0, cls_name="plate", conf=0.9, xyxy=[0, 0, 10, 10]),
            Det(cls_id=0, cls_name="plate", conf=0.5, xyxy=[50, 50, 60, 60]),
        ]
        assert len(nms_same_class(dets, iou_thr=0.45)) == 2

    @pytest.mark.unit
    def test_normalises_names_when_comparing(self):
        dets = [
            Det(cls_id=0, cls_name="No-Helmet", conf=0.9, xyxy=[0, 0, 10, 10]),
            Det(cls_id=0, cls_name="no helmet", conf=0.5, xyxy=[1, 1, 11, 11]),
        ]
        assert len(nms_same_class(dets, iou_thr=0.45)) == 1


class TestHelpers:
    @pytest.mark.unit
    def test_norm_name(self):
        assert norm_name("No-Helmet ") == "no_helmet"
        assert norm_name("License Plate") == "license_plate"

    @pytest.mark.unit
    def test_expand_box(self):
        # box 10x10 at (10,10)-(20,20), 50% pad each way on x, none on y
        assert expand_box([10, 10, 20, 20], w=100, h=100, xpad=0.5, ypad=0.0) == [5, 10, 25, 20]

    @pytest.mark.unit
    def test_crop_box_returns_pixels_and_offsets(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, box = crop_box(img, [10, 10, 20, 20])
        assert crop.shape == (10, 10, 3)
        assert box == [10, 10, 20, 20]

    @pytest.mark.unit
    def test_offset_det(self):
        det = Det(cls_id=1, cls_name="helmet", conf=0.8, xyxy=[5, 5, 10, 10])
        shifted = offset_det(det, [100, 200])
        assert shifted.xyxy == [105, 205, 110, 210]
        renamed = offset_det(det, [100, 200], new_name="no_helmet")
        assert renamed.cls_name == "no_helmet"
