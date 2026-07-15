from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from walker_gait.core.types import ImuSample
from walker_gait.core.factory import Registry

IMU_REGISTRY = Registry("imu_source")


class ImuSource(ABC):
    """Camera-independent stream of IMU samples. Develop and test entirely in
    parallel with the vision pipeline -- only loosely coupled via timestamps
    at the fusion/cross-validation stage (see gait/imu_events.py)."""

    @abstractmethod
    def get_reading(self) -> Optional[ImuSample]:
        raise NotImplementedError

    def close(self) -> None:
        pass
