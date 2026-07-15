"""
Two zero/low-dependency detector backends for testing the pipeline before
YOLOv8 weights are downloaded and tuned:

- `dummy`: always returns a fixed, centered bounding box. Useful for
  synthetic-source testing (SyntheticSource always centers its person) and
  for exercising the tracker/pose2d/depth stages without any detection logic
  in the loop at all.
- `motion`: simple background-subtraction blob detector (OpenCV MOG2).
  Not a person detector -- it'll box up anything that moves -- but it's
  enough to validate the detector->tracker->pose2d chain against REAL
  webcam frames before YOLOv8 is wired in, since it needs only opencv
  (already a core dependency), no model weights.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from walker_gait.core.types import Detection
from walker_gait.detector.base import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("dummy")
class DummyDetector(Detector):
    def __init__(self, bbox_fraction: tuple = (0.25, 0.05, 0.75, 0.98), confidence: float = 0.95):
        self.bbox_fraction = bbox_fraction
        self.confidence = confidence

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        h, w = rgb.shape[:2]
        fx1, fy1, fx2, fy2 = self.bbox_fraction
        bbox = (w * fx1, h * fy1, w * fx2, h * fy2)
        return [Detection(bbox=bbox, confidence=self.confidence)]


@DETECTOR_REGISTRY.register("motion")
class MotionDetector(Detector):
    def __init__(self, min_area_fraction: float = 0.02, history: int = 500,
                 var_threshold: float = 16.0):
        import cv2
        self._cv2 = cv2
        self.min_area_fraction = min_area_fraction
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        cv2 = self._cv2
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = self.subtractor.apply(bgr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area_fraction * (w * h):
            return []

        x, y, bw, bh = cv2.boundingRect(largest)
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        # No real confidence score from background subtraction -- report a
        # flat placeholder so downstream confidence-threshold logic still works.
        return [Detection(bbox=bbox, confidence=0.5)]
