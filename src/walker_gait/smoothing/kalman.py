"""
Per-joint constant-velocity Kalman filter: the LIVE smoothing tier.

Why this instead of MotionBERT for the live path: MotionBERT needs a temporal
window (future context) to do its best work, which adds latency that's a
problem for a real-time clinical UI. A causal Kalman filter runs in
microseconds, requires no future frames, and naturally "coasts" through brief
occlusions (e.g. the walker frame blocking an ankle) by predicting forward
from the last known velocity until a new measurement arrives.

State per joint: [x, y, z, vx, vy, vz]. Independent filter per joint (18
scalar filters total for 13 joints... actually 6 x 13 = 78 state scalars,
still trivial compute).
"""
from __future__ import annotations
import numpy as np

from walker_gait.core.types import Skeleton3D, NUM_JOINTS
from walker_gait.smoothing.base import SkeletonSmoother, SMOOTHER_REGISTRY


class _JointKalman:
    def __init__(self, dt: float, process_noise: float, measurement_noise: float):
        self.dt = dt
        # State transition: constant velocity model
        self.F = np.eye(6, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = dt
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)

        q = process_noise
        self.Q = np.eye(6, dtype=np.float64) * q
        self.R = np.eye(3, dtype=np.float64) * measurement_noise

        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_meas: np.ndarray):
        z = z_meas.reshape(3, 1)
        if not self.initialized:
            self.x[:3, 0] = z_meas
            self.initialized = True
            return
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3, 0].astype(np.float32)


@SMOOTHER_REGISTRY.register("kalman")
class KalmanSkeletonSmoother(SkeletonSmoother):
    def __init__(self, dt: float = 1.0 / 30.0, process_noise: float = 0.01,
                 measurement_noise: float = 0.02, coast_frames_max: int = 10):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.coast_frames_max = coast_frames_max
        self._filters = [None] * NUM_JOINTS
        self._coast_count = [0] * NUM_JOINTS

    def reset(self) -> None:
        self._filters = [None] * NUM_JOINTS
        self._coast_count = [0] * NUM_JOINTS

    def update(self, skeleton3d: Skeleton3D) -> Skeleton3D:
        out_pts = np.zeros_like(skeleton3d.keypoints)
        out_valid = np.zeros_like(skeleton3d.valid)

        for j in range(NUM_JOINTS):
            if self._filters[j] is None:
                self._filters[j] = _JointKalman(self.dt, self.process_noise, self.measurement_noise)

            kf = self._filters[j]
            kf.predict()

            if skeleton3d.valid[j]:
                kf.update(skeleton3d.keypoints[j])
                self._coast_count[j] = 0
                out_valid[j] = True
            else:
                # Occlusion / missing depth: coast on the predicted state for
                # a bounded number of frames, then mark invalid so downstream
                # gait-event logic doesn't trust a stale prediction forever.
                self._coast_count[j] += 1
                out_valid[j] = self._coast_count[j] <= self.coast_frames_max and kf.initialized

            out_pts[j] = kf.position

        return Skeleton3D(keypoints=out_pts, valid=out_valid,
                           track_id=skeleton3d.track_id, timestamp=skeleton3d.timestamp)
