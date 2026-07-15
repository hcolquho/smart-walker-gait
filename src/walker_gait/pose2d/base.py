from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from walker_gait.core.types import Skeleton2D, TrackedDetection
from walker_gait.core.factory import Registry

POSE2D_REGISTRY = Registry("pose2d")


class PoseEstimator2D(ABC):
    """Input: RGB frame + a person bounding box. Output: per-joint (u,v) pixel
    coordinates + confidence. Config-swappable: RTMPose for live/low-latency,
    ViTPose for offline/high-fidelity re-processing -- same interface."""

    @abstractmethod
    def estimate(self, rgb: np.ndarray, detection: TrackedDetection, timestamp: float) -> Skeleton2D:
        raise NotImplementedError
