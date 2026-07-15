"""
Real person detector using Ultralytics YOLOv8. Lazy-imports `ultralytics` so
every other stage in this package stays dependency-free until you're ready
for this one.

Install:
    pip install ultralytics

First run will auto-download the chosen weights file (e.g. `yolov8n.pt`,
~6MB) from the Ultralytics release assets on first use if it's not already
present locally -- point `weights` at a local path once you've fine-tuned or
otherwise pinned a specific checkpoint.

COCO class_id 0 = "person" (Ultralytics uses standard COCO-80 class indices),
which is what `person_class_id` defaults to and filters on.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from walker_gait.core.types import Detection
from walker_gait.detector.base import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register("yolov8")
class Yolov8Detector(Detector):
    def __init__(self,
                 weights: str = "yolov8n.pt",
                 conf_threshold: float = 0.4,
                 iou_threshold: float = 0.45,
                 person_class_id: int = 0,
                 device: str = "cpu",
                 single_person_mode: bool = True,
                 imgsz: int = 640):
        """
        single_person_mode: this pipeline is built for one patient using the
            walker at a time. When True, only the single highest-confidence
            person detection is returned per frame -- this keeps the IOU
            tracker from spawning spurious extra tracks if a bystander walks
            through frame. Set False if you need multi-person detections for
            some other analysis.
        """
        from ultralytics import YOLO  # lazy import, heavy dependency
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.device = device
        self.single_person_mode = single_person_mode
        self.imgsz = imgsz

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: List[Detection] = []
        if results:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                detections.append(Detection(bbox=tuple(xyxy), confidence=conf,
                                             class_id=self.person_class_id))

        if self.single_person_mode and len(detections) > 1:
            detections = [max(detections, key=lambda d: d.confidence)]

        return detections
