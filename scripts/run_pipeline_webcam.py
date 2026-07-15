"""
Runs Detector -> Tracker -> (mock 2D pose) live against your actual webcam,
drawing the bbox/track_id/keypoints overlay so you can visually confirm the
detector+tracker chain works on real video BEFORE the Femto Bolt (and before
YOLOv8 weights are downloaded/tuned).

Two detector choices, no ML weights required for either:
    --detector dummy   fixed centered box (fine for a controlled test rig
                        where you always stand centered in frame)
    --detector motion  background-subtraction blob detector (reacts to
                        actual movement -- better sanity check of the
                        detector->tracker->pose2d wiring on real footage)

Once ultralytics is installed and yolov8n.pt is downloaded, add
"yolov8" as a third option by swapping in Yolov8Detector the same way.

Usage:
    PYTHONPATH=src python scripts/run_pipeline_webcam.py --detector motion
Press 'q' to quit.
"""
from __future__ import annotations
import argparse
import time

import cv2
import numpy as np

from walker_gait.frame_source.webcam import WebcamSource
from walker_gait.detector.dummy import DummyDetector, MotionDetector
from walker_gait.tracking.iou_tracker import IouTracker
from walker_gait.pose2d.dummy import DummyPoseEstimator2D
from walker_gait.core.types import JOINT_NAMES


def build_detector(name: str):
    if name == "dummy":
        return DummyDetector()
    if name == "motion":
        return MotionDetector()
    raise ValueError(f"Unknown detector '{name}'. Use 'dummy' or 'motion' "
                      f"(add 'yolov8' yourself once ultralytics is installed).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="motion", choices=["dummy", "motion"])
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    src = WebcamSource(device_index=args.camera_index)
    detector = build_detector(args.detector)
    tracker = IouTracker()
    pose_estimator = DummyPoseEstimator2D()

    print("Press 'q' to quit.")
    try:
        for frame in src:
            detections = detector.detect(frame.rgb)
            tracked = tracker.update(detections, timestamp=frame.timestamp)

            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            for td in tracked:
                x1, y1, x2, y2 = [int(v) for v in td.bbox]
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f"id={td.track_id} conf={td.confidence:.2f}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                skeleton2d = pose_estimator.estimate(frame.rgb, td, frame.timestamp)
                for j, name in enumerate(JOINT_NAMES):
                    u, v = skeleton2d.keypoints[j]
                    cv2.circle(bgr, (int(u), int(v)), 3, (0, 128, 255), -1)

            cv2.imshow("Detector -> Tracker -> (mock) Pose2D", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
