"""
Runs the full pipeline end-to-end using only synthetic data and mock/lightweight
backends -- no camera, no GPU, no heavy ML installs required. This is the
script to run to sanity-check the whole architecture before the Femto Bolt
arrives, and the reference for how the stages wire together.

Usage:
    cd smart_walker_gait
    pip install -r requirements.txt
    PYTHONPATH=src python scripts/run_pipeline_demo.py
"""
from __future__ import annotations
import numpy as np

from walker_gait.frame_source.synthetic import SyntheticSource
from walker_gait.core.types import Detection, TrackedDetection, Skeleton2D, NUM_JOINTS
from walker_gait.tracking.iou_tracker import IouTracker
from walker_gait.depth.backprojection import DepthBackprojector
from walker_gait.smoothing.kalman import KalmanSkeletonSmoother
from walker_gait.gait.events import VerticalVelocityEventDetector
from walker_gait.gait.metrics import GaitMetricsCalculator


def main():
    print("=== Smart Walker Gait Pipeline: synthetic end-to-end demo ===\n")

    src = SyntheticSource(fps=30.0, duration_s=8.0, cadence_steps_per_min=100.0,
                           left_right_asymmetry=0.2, noise_std_px=1.5)
    tracker = IouTracker()
    backprojector = DepthBackprojector(window=5)
    smoother = KalmanSkeletonSmoother(dt=1 / 30.0)
    event_detector = VerticalVelocityEventDetector(fps=30.0)

    smoothed_skeletons = []

    for frame in src:
        # --- Detection stage stand-in: synthetic person always centered ---
        # (In the real pipeline this comes from your existing YOLOv8 detector.)
        h, w = frame.rgb.shape[:2]
        detection = Detection(bbox=(w * 0.25, h * 0.05, w * 0.75, h * 0.98), confidence=0.95)

        # --- Tracking ---
        tracked = tracker.update([detection], timestamp=frame.timestamp)
        if not tracked:
            continue
        td: TrackedDetection = tracked[0]

        # --- 2D pose (using the synthetic ground-truth-with-noise keypoints
        #     as a stand-in for a real 2D pose model -- see pose2d/dummy.py) ---
        skeleton2d = Skeleton2D(keypoints=frame.gt_2d_noisy,
                                 scores=np.ones(NUM_JOINTS),
                                 track_id=td.track_id, timestamp=frame.timestamp)

        # --- Depth backprojection ---
        skeleton3d = backprojector.backproject_skeleton(skeleton2d, frame.depth, frame.intrinsics)

        # --- Live smoothing (Kalman) ---
        smoothed = smoother.update(skeleton3d)
        smoothed_skeletons.append(smoothed)

    # --- Gait events + metrics (batch, over the buffered session) ---
    events = event_detector.process_sequence(smoothed_skeletons)
    metrics = GaitMetricsCalculator().compute(events, skeletons=smoothed_skeletons)

    print(f"Processed {len(smoothed_skeletons)} frames, detected {len(events)} gait events.\n")
    print("Gait metrics:")
    for field in ["cadence_steps_per_min", "stride_time_left_s", "stride_time_right_s",
                  "step_time_asymmetry", "step_length_asymmetry", "loading_asymmetry",
                  "double_support_time_s", "aggregate_asymmetry_score"]:
        print(f"  {field:32s} {getattr(metrics, field):.4f}")
    print(f"\n  (injected ground-truth asymmetry was 0.20 -- compare against "
          f"step_time_asymmetry above as a sanity check)")


if __name__ == "__main__":
    main()
