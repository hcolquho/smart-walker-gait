import numpy as np
from walker_gait.detector.dummy import DummyDetector, MotionDetector


def test_dummy_detector_returns_centered_bbox_scaled_to_frame():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    det = DummyDetector()
    detections = det.detect(rgb)

    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].bbox
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480
    assert detections[0].confidence > 0


def test_dummy_detector_scales_with_different_frame_sizes():
    det = DummyDetector()
    small = det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    large = det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert small[0].bbox[2] < large[0].bbox[2], "bbox should scale with frame width"


def test_motion_detector_finds_no_detections_on_static_frames():
    det = MotionDetector()
    static_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    # feed several identical frames to let the background model converge
    for _ in range(10):
        detections = det.detect(static_frame)
    assert detections == []


def test_motion_detector_finds_a_moving_blob():
    det = MotionDetector(min_area_fraction=0.01)
    h, w = 480, 640
    background = np.full((h, w, 3), 100, dtype=np.uint8)

    # establish background model
    for _ in range(15):
        det.detect(background)

    # introduce a clearly different bright rectangular "person" blob
    moving_frame = background.copy()
    moving_frame[150:400, 250:400] = 250

    detections = det.detect(moving_frame)
    assert len(detections) >= 1
    x1, y1, x2, y2 = detections[0].bbox
    # detected box should roughly overlap the injected blob region
    assert x1 < 400 and x2 > 250 and y1 < 400 and y2 > 150
