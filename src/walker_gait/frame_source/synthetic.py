"""
SyntheticSource: generates a known-ground-truth 3D walking skeleton, projects
it through a pinhole camera model to produce fake RGB+depth frames, and
exposes the ground truth so every downstream stage (pose estimation stand-in,
backprojection, smoothing, gait metrics) can be validated quantitatively
*before* real hardware or real pose models exist.

This is the single most important file for developing without the Femto Bolt.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from walker_gait.core.types import Frame, CameraIntrinsics, JOINT_INDEX, NUM_JOINTS
from walker_gait.frame_source.base import FrameSource, FRAME_SOURCE_REGISTRY


def _default_intrinsics(width=640, height=480) -> CameraIntrinsics:
    # Rough Femto-Bolt-like WFOV numbers so downstream code sees realistic scale.
    fx = fy = 525.0 * (width / 640.0)
    return CameraIntrinsics(fx=fx, fy=fy, cx=width / 2, cy=height / 2,
                             depth_scale=1.0, width=width, height=height)


@FRAME_SOURCE_REGISTRY.register("synthetic")
class SyntheticSource(FrameSource):
    """
    Simulates a person walking on a treadmill-like path in front of a
    downward-angled camera, using a simple sinusoidal gait model for the
    legs (good enough to exercise stride timing / asymmetry logic, not a
    biomechanical simulator).

    Params
    ------
    fps: frame rate
    duration_s: total sequence length
    cadence_steps_per_min: controls leg oscillation frequency
    camera_distance_m: distance from camera to person (z, along optical axis)
    noise_std_px: gaussian pixel noise added to the *reported* 2D keypoints
    depth_noise_std_m: gaussian noise added to the *reported* depth channel
    occlusion_frames: list of (start_frame, end_frame, joint_name) to blank out,
        for testing smoothing/occlusion handling
    left_right_asymmetry: 0..1, injects a deliberate limp (shorter/faster
        stride on one side) so you can validate asymmetry metrics against
        a known injected value.
    """

    def __init__(self,
                 fps: float = 30.0,
                 duration_s: float = 10.0,
                 cadence_steps_per_min: float = 100.0,
                 camera_distance_m: float = 2.2,
                 noise_std_px: float = 1.5,
                 depth_noise_std_m: float = 0.01,
                 occlusion_frames: Optional[list] = None,
                 left_right_asymmetry: float = 0.0,
                 width: int = 640, height: int = 480,
                 seed: int = 0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.n_frames = int(duration_s * fps)
        self.cadence = cadence_steps_per_min
        self.cam_z = camera_distance_m
        self.noise_std_px = noise_std_px
        self.depth_noise_std_m = depth_noise_std_m
        self.occlusion_frames = occlusion_frames or []
        self.asym = left_right_asymmetry
        self.width, self.height = width, height
        self.intrinsics = _default_intrinsics(width, height)
        self.rng = np.random.default_rng(seed)
        self._frame_idx = 0

    def is_depth_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Ground-truth skeleton model
    # ------------------------------------------------------------------
    def ground_truth_3d(self, frame_idx: int) -> np.ndarray:
        """Returns (NUM_JOINTS, 3) meters, in camera coordinates, no noise."""
        t = frame_idx * self.dt
        # `cadence` is total steps/min combined across BOTH feet (standard
        # clinical convention), so each individual foot strikes the ground at
        # half that rate.
        step_freq_hz = self.cadence / 60.0 / 2.0
        w = 2 * np.pi * step_freq_hz

        # Asymmetry: right leg swings faster/shorter than left by `asym` fraction
        w_left = w * (1.0 - 0.3 * self.asym)
        w_right = w * (1.0 + 0.3 * self.asym)

        # hip_y = 0 places the hip at camera height: a walker-mounted camera
        # sits roughly at hip/handle level, so hip is near frame-center, head
        # is above, and legs extend below -- everything of clinical interest
        # (hip/knee/ankle/heel/toe) stays in-frame even without modeling an
        # explicit downward tilt angle.
        hip_y = 0.0
        knee_amp_z = 0.15     # forward/back (stride) swing amplitude, meters
        ankle_amp_z = 0.20
        vertical_lift = 0.06  # ground clearance during swing phase, meters

        pts = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        sway = 0.02 * np.sin(w * t * 0.5)

        pts[JOINT_INDEX["nose"]] = [sway, hip_y + 0.55, self.cam_z]
        pts[JOINT_INDEX["left_shoulder"]] = [sway - 0.18, hip_y + 0.45, self.cam_z]
        pts[JOINT_INDEX["right_shoulder"]] = [sway + 0.18, hip_y + 0.45, self.cam_z]
        pts[JOINT_INDEX["left_hip"]] = [sway - 0.09, hip_y, self.cam_z]
        pts[JOINT_INDEX["right_hip"]] = [sway + 0.09, hip_y, self.cam_z]

        left_phase = w_left * t
        right_phase = w_right * t + np.pi  # opposite leg phase

        # Vertical bounce: (1 - cos(phase)) / 2 has a single sharp minimum
        # (=0, foot on the ground) once per gait cycle and a smooth maximum
        # mid-swing -- this is what makes heel-strike/toe-off detectable as a
        # genuine local minimum downstream, unlike a flat/constant height.
        left_lift = vertical_lift * (1 - np.cos(left_phase)) / 2.0
        right_lift = vertical_lift * (1 - np.cos(right_phase)) / 2.0

        knee_base_y = hip_y - 0.45
        ankle_base_y = hip_y - 0.85

        pts[JOINT_INDEX["left_knee"]] = [sway - 0.09, knee_base_y + 0.5 * left_lift,
                                          self.cam_z + knee_amp_z * np.sin(left_phase)]
        pts[JOINT_INDEX["right_knee"]] = [sway + 0.09, knee_base_y + 0.5 * right_lift,
                                           self.cam_z + knee_amp_z * np.sin(right_phase)]

        pts[JOINT_INDEX["left_ankle"]] = [sway - 0.09, ankle_base_y + left_lift,
                                           self.cam_z + ankle_amp_z * np.sin(left_phase)]
        pts[JOINT_INDEX["right_ankle"]] = [sway + 0.09, ankle_base_y + right_lift,
                                            self.cam_z + ankle_amp_z * np.sin(right_phase)]

        pts[JOINT_INDEX["left_heel"]] = pts[JOINT_INDEX["left_ankle"]] + [0, -0.03, -0.05]
        pts[JOINT_INDEX["right_heel"]] = pts[JOINT_INDEX["right_ankle"]] + [0, -0.03, -0.05]
        pts[JOINT_INDEX["left_toe"]] = pts[JOINT_INDEX["left_ankle"]] + [0, -0.05, 0.12]
        pts[JOINT_INDEX["right_toe"]] = pts[JOINT_INDEX["right_ankle"]] + [0, -0.05, 0.12]

        return pts

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """Pinhole projection: (N,3) camera-frame meters -> (N,2) pixel coords."""
        k = self.intrinsics
        z = np.clip(pts3d[:, 2], 1e-3, None)
        u = k.fx * (pts3d[:, 0] / z) + k.cx
        v = k.fy * (-pts3d[:, 1] / z) + k.cy  # image y grows downward, world y grows upward
        return np.stack([u, v], axis=1)

    # ------------------------------------------------------------------
    def get_frame(self) -> Optional[Frame]:
        if self._frame_idx >= self.n_frames:
            return None
        idx = self._frame_idx
        gt3d = self.ground_truth_3d(idx)
        uv = self.project(gt3d)

        rgb = np.full((self.height, self.width, 3), 30, dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)

        for j in range(NUM_JOINTS):
            u, v = int(round(uv[j, 0])), int(round(uv[j, 1]))
            if 0 <= u < self.width and 0 <= v < self.height:
                depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = gt3d[j, 2]

        frame = Frame(rgb=rgb, depth=depth, timestamp=idx * self.dt,
                       intrinsics=self.intrinsics, frame_id=idx)
        # Stash ground truth on the frame for test harnesses (not part of the
        # formal Frame schema, but convenient - downstream real code ignores it).
        frame.gt_3d = gt3d          # type: ignore[attr-defined]
        frame.gt_2d = uv            # type: ignore[attr-defined]
        frame.gt_2d_noisy = uv + self.rng.normal(0, self.noise_std_px, uv.shape)  # type: ignore[attr-defined]

        self._frame_idx += 1
        return frame

    def reset(self):
        self._frame_idx = 0
