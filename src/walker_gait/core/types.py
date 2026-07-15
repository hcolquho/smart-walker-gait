"""
Shared data types used across every stage of the pipeline:
FrameSource -> Detector -> Tracker -> PoseEstimator2D -> Depth -> Smoother -> Gait

Keeping these in one place means every stage speaks the same language and
backends can be swapped (mock <-> real) without touching neighboring stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Joint naming: fixed vocabulary so every stage indexes joints the same way.
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_toe", "right_toe",
    "left_heel", "right_heel",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
NUM_JOINTS = len(JOINT_NAMES)


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class CameraIntrinsics:
    """Pinhole camera model parameters. Swappable per-sensor via YAML."""
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0  # multiply raw depth reading by this to get meters
    width: int = 640
    height: int = 480

    @staticmethod
    def from_dict(d: dict) -> "CameraIntrinsics":
        return CameraIntrinsics(**d)

    def to_dict(self) -> dict:
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                    depth_scale=self.depth_scale, width=self.width, height=self.height)


@dataclass
class Frame:
    """One synchronized capture from the sensor layer."""
    rgb: np.ndarray                      # HxWx3 uint8
    timestamp: float                     # seconds
    depth: Optional[np.ndarray] = None   # HxW float32, meters (None if unavailable)
    intrinsics: Optional[CameraIntrinsics] = None
    frame_id: int = 0


@dataclass
class Detection:
    """Person bounding box from the detector stage."""
    bbox: tuple  # (x1, y1, x2, y2) pixels
    confidence: float
    class_id: int = 0


@dataclass
class TrackedDetection(Detection):
    track_id: int = -1


@dataclass
class Skeleton2D:
    """(u, v) pixel keypoints for one person, one frame."""
    keypoints: np.ndarray   # (NUM_JOINTS, 2) float32, pixel coords
    scores: np.ndarray      # (NUM_JOINTS,) float32, confidence per joint
    track_id: int = -1
    timestamp: float = 0.0


@dataclass
class Skeleton3D:
    """3D joints in camera coordinate frame (meters), one frame."""
    keypoints: np.ndarray   # (NUM_JOINTS, 3) float32
    valid: np.ndarray       # (NUM_JOINTS,) bool - False where depth lookup failed
    track_id: int = -1
    timestamp: float = 0.0


@dataclass
class ImuSample:
    accel: np.ndarray       # (3,) m/s^2
    gyro: np.ndarray        # (3,) rad/s
    quat: np.ndarray        # (4,) w,x,y,z orientation
    timestamp: float = 0.0


class GaitEventType(str, Enum):
    HEEL_STRIKE = "heel_strike"
    TOE_OFF = "toe_off"


@dataclass
class GaitEvent:
    type: GaitEventType
    side: Side
    timestamp: float
    frame_id: int = -1


@dataclass
class GaitMetrics:
    cadence_steps_per_min: float = 0.0
    stride_time_left_s: float = 0.0
    stride_time_right_s: float = 0.0
    step_time_asymmetry: float = 0.0
    step_length_asymmetry: float = 0.0
    loading_asymmetry: float = 0.0
    double_support_time_s: float = 0.0
    aggregate_asymmetry_score: float = 0.0
    extra: dict = field(default_factory=dict)
