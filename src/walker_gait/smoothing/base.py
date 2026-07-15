from __future__ import annotations
from abc import ABC, abstractmethod

from walker_gait.core.types import Skeleton3D
from walker_gait.core.factory import Registry

SMOOTHER_REGISTRY = Registry("smoother")


class SkeletonSmoother(ABC):
    """Two-tier design:
    - 'kalman' backend: causal, per-joint constant-velocity Kalman filter.
      Zero added latency -> used in the LIVE UI path.
    - 'motionbert' backend: transformer over a frame window, higher fidelity,
      adds latency -> used in the OFFLINE / clinical-report path only.
    Both implement this same interface so the gait-metrics stage doesn't care
    which one produced the skeleton it's consuming.
    """

    @abstractmethod
    def update(self, skeleton3d: Skeleton3D) -> Skeleton3D:
        """Causal / streaming smoothers: consume one frame, return one frame."""
        raise NotImplementedError

    def reset(self) -> None:
        pass
