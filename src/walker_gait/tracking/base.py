from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from walker_gait.core.types import Detection, TrackedDetection
from walker_gait.core.factory import Registry

TRACKER_REGISTRY = Registry("tracker")


class Tracker(ABC):
    """Assigns a persistent track_id to detections across frames. Single-
    person use case (patient on the walker) so this only needs to bridge
    brief false-negative/occlusion gaps, not full multi-object association."""

    @abstractmethod
    def update(self, detections: List[Detection], timestamp: float) -> List[TrackedDetection]:
        raise NotImplementedError

    def reset(self) -> None:
        pass
