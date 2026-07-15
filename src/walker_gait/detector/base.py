from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from walker_gait.core.types import Detection
from walker_gait.core.factory import Registry

DETECTOR_REGISTRY = Registry("detector")


class Detector(ABC):
    """Input: one RGB frame. Output: list of person Detection boxes (before
    tracking assigns track_id). Config-swappable via DETECTOR_REGISTRY --
    everything downstream (tracker, pose2d, ...) only depends on
    `core.types.Detection`, never on a concrete detector implementation."""

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> List[Detection]:
        raise NotImplementedError
