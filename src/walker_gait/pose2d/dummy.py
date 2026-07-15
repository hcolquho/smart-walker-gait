"""
DummyPoseEstimator2D: lets you run and test the FULL pipeline end-to-end with
zero ML dependencies installed. Two modes:

- If the Frame carries synthetic ground-truth 2D keypoints (set by
  SyntheticSource as `frame.gt_2d_noisy`), just return those -- perfect for
  validating depth/smoothing/gait-metrics stages against known truth.
- Otherwise (e.g. real webcam frames, no ground truth available), returns a
  crude anthropometric keypoint layout scaled to the detection bbox, with
  random jitter -- enough to exercise the plumbing (shapes, timing, UI
  rendering) before RTMPose is installed and wired in.
"""
from __future__ import annotations
import numpy as np

from walker_gait.core.types import Skeleton2D, TrackedDetection, NUM_JOINTS, JOINT_INDEX
from walker_gait.pose2d.base import PoseEstimator2D, POSE2D_REGISTRY

# Fractional (x, y) position of each joint within the bbox, top-left origin,
# roughly matching a standing/walking person silhouette.
_BBOX_FRACTIONS = {
    "nose": (0.5, 0.05),
    "left_shoulder": (0.35, 0.18), "right_shoulder": (0.65, 0.18),
    "left_hip": (0.42, 0.50), "right_hip": (0.58, 0.50),
    "left_knee": (0.42, 0.72), "right_knee": (0.58, 0.72),
    "left_ankle": (0.42, 0.95), "right_ankle": (0.58, 0.95),
    "left_heel": (0.40, 0.97), "right_heel": (0.60, 0.97),
    "left_toe": (0.44, 0.99), "right_toe": (0.56, 0.99),
}


@POSE2D_REGISTRY.register("dummy")
class DummyPoseEstimator2D(PoseEstimator2D):
    def __init__(self, jitter_px: float = 2.0, seed: int = 0):
        self.jitter_px = jitter_px
        self.rng = np.random.default_rng(seed)

    def estimate(self, rgb: np.ndarray, detection: TrackedDetection, timestamp: float) -> Skeleton2D:
        keypoints = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        scores = np.ones((NUM_JOINTS,), dtype=np.float32) * 0.9

        gt_2d_noisy = getattr(rgb, "gt_2d_noisy", None)  # only present if caller passed a Frame-like obj
        if gt_2d_noisy is not None:
            keypoints = np.asarray(gt_2d_noisy, dtype=np.float32)
        else:
            x1, y1, x2, y2 = detection.bbox
            w, h = (x2 - x1), (y2 - y1)
            for name, (fx, fy) in _BBOX_FRACTIONS.items():
                idx = JOINT_INDEX[name]
                keypoints[idx] = [x1 + fx * w, y1 + fy * h]
            keypoints += self.rng.normal(0, self.jitter_px, keypoints.shape)

        return Skeleton2D(keypoints=keypoints, scores=scores,
                           track_id=detection.track_id, timestamp=timestamp)
