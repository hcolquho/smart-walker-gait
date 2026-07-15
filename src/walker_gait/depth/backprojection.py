"""
Depth lookup + backprojection: (u, v, depth_map) -> (x, y, z) meters.

Median-of-neighborhood sampling (rather than single-pixel reads) mitigates
depth noise at joint boundaries/edges, which is the dominant noise source on
ToF sensors like the Femto Bolt. Optional segmentation-mask-aware sampling
further restricts the window to pixels that are plausibly part of the person.

This module has NO hardware dependency -- fully testable today against
SyntheticSource's known depth map.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from walker_gait.core.types import Skeleton2D, Skeleton3D, CameraIntrinsics, NUM_JOINTS


class DepthBackprojector:
    def __init__(self, window: int = 5, min_valid_fraction: float = 0.3,
                 max_depth_m: float = 6.0, min_depth_m: float = 0.2):
        """
        window: side length of the square neighborhood (must be odd) used for
            median depth sampling around each (u, v) keypoint.
        min_valid_fraction: if fewer than this fraction of pixels in the
            window have plausible depth (nonzero, within [min,max]), mark the
            joint invalid rather than returning a garbage estimate.
        """
        assert window % 2 == 1, "window must be odd so it's centered on the pixel"
        self.window = window
        self.min_valid_fraction = min_valid_fraction
        self.max_depth_m = max_depth_m
        self.min_depth_m = min_depth_m

    def _sample_depth(self, depth_map: np.ndarray, u: int, v: int,
                       mask: Optional[np.ndarray]) -> Optional[float]:
        h, w = depth_map.shape
        r = self.window // 2
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        if u0 >= u1 or v0 >= v1:
            return None

        patch = depth_map[v0:v1, u0:u1]
        valid = (patch > self.min_depth_m) & (patch < self.max_depth_m)
        if mask is not None:
            mask_patch = mask[v0:v1, u0:u1]
            valid &= mask_patch.astype(bool)

        n_total = patch.size
        n_valid = int(valid.sum())
        if n_total == 0 or (n_valid / n_total) < self.min_valid_fraction:
            return None

        return float(np.median(patch[valid]))

    def backproject_joint(self, u: float, v: float, z: float,
                           intrinsics: CameraIntrinsics) -> np.ndarray:
        x = (u - intrinsics.cx) * z / intrinsics.fx
        y = -(v - intrinsics.cy) * z / intrinsics.fy  # flip so +y is "up" in world/camera frame
        return np.array([x, y, z], dtype=np.float32)

    def backproject_skeleton(self, skeleton2d: Skeleton2D, depth_map: np.ndarray,
                              intrinsics: CameraIntrinsics,
                              mask: Optional[np.ndarray] = None) -> Skeleton3D:
        pts3d = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        valid = np.zeros((NUM_JOINTS,), dtype=bool)

        depth_scaled = depth_map * intrinsics.depth_scale

        for j in range(NUM_JOINTS):
            u, v = skeleton2d.keypoints[j]
            u_i, v_i = int(round(u)), int(round(v))
            z = self._sample_depth(depth_scaled, u_i, v_i, mask)
            if z is None:
                continue
            pts3d[j] = self.backproject_joint(u, v, z, intrinsics)
            valid[j] = True

        return Skeleton3D(keypoints=pts3d, valid=valid,
                           track_id=skeleton2d.track_id, timestamp=skeleton2d.timestamp)
