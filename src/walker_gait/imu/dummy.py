from __future__ import annotations
import time
from typing import Optional
import numpy as np

from walker_gait.core.types import ImuSample
from walker_gait.imu.base import ImuSource, IMU_REGISTRY


@IMU_REGISTRY.register("dummy")
class DummyImuSource(ImuSource):
    """Generates plausible IMU samples for testing the fusion/cross-check
    logic without real BNO085 hardware attached."""

    def __init__(self, cadence_steps_per_min: float = 100.0, seed: int = 0):
        self.w = 2 * np.pi * (cadence_steps_per_min / 60.0)
        self.rng = np.random.default_rng(seed)
        self._t0 = time.time()

    def get_reading(self) -> Optional[ImuSample]:
        t = time.time() - self._t0
        accel = np.array([0.0, 9.81 + 0.5 * np.sin(self.w * t), 0.3 * np.cos(self.w * t)],
                          dtype=np.float32) + self.rng.normal(0, 0.02, 3)
        gyro = np.array([0.0, 0.0, 0.1 * np.sin(self.w * t)], dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return ImuSample(accel=accel, gyro=gyro, quat=quat, timestamp=time.time())
