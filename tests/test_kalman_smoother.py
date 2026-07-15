import numpy as np
from walker_gait.core.types import Skeleton3D, NUM_JOINTS
from walker_gait.smoothing.kalman import KalmanSkeletonSmoother


def _make_skeleton(pos, valid=True, t=0.0):
    pts = np.tile(np.array(pos, dtype=np.float32), (NUM_JOINTS, 1))
    return Skeleton3D(keypoints=pts, valid=np.full(NUM_JOINTS, valid), timestamp=t)


def test_kalman_reduces_gaussian_measurement_noise():
    rng = np.random.default_rng(0)
    smoother = KalmanSkeletonSmoother(dt=1 / 30.0, process_noise=0.001, measurement_noise=0.02)

    true_pos = np.array([0.0, 1.0, 1.8])
    raw_errors, smoothed_errors = [], []

    for i in range(60):
        noisy_pos = true_pos + rng.normal(0, 0.03, 3)
        skel = _make_skeleton(noisy_pos, valid=True, t=i / 30.0)
        smoothed = smoother.update(skel)
        if i > 10:  # let the filter converge past initial transient
            raw_errors.append(np.linalg.norm(noisy_pos - true_pos))
            smoothed_errors.append(np.linalg.norm(smoothed.keypoints[0] - true_pos))

    assert np.mean(smoothed_errors) < np.mean(raw_errors), \
        "Kalman-smoothed error should be lower than raw measurement error"


def test_kalman_coasts_through_short_occlusion():
    smoother = KalmanSkeletonSmoother(dt=1 / 30.0, coast_frames_max=10)

    # establish a moving joint so velocity state is non-trivial
    for i in range(10):
        pos = [0.0, 1.0 + 0.01 * i, 1.8]
        smoother.update(_make_skeleton(pos, valid=True, t=i / 30.0))

    # occlude for 5 frames (within coast_frames_max=10)
    last_valid_out = None
    for i in range(10, 15):
        out = smoother.update(_make_skeleton([0, 0, 0], valid=False, t=i / 30.0))
        last_valid_out = out

    assert last_valid_out.valid[0], "should still be marked valid while coasting within budget"
    # position should have kept moving in the established direction, not frozen or reset to zero
    assert last_valid_out.keypoints[0, 1] > 1.05


def test_kalman_marks_invalid_after_coast_budget_exceeded():
    smoother = KalmanSkeletonSmoother(dt=1 / 30.0, coast_frames_max=3)
    smoother.update(_make_skeleton([0, 1, 1.8], valid=True, t=0.0))

    out = None
    for i in range(1, 10):  # exceeds coast_frames_max=3
        out = smoother.update(_make_skeleton([0, 0, 0], valid=False, t=i / 30.0))

    assert not out.valid[0], "joint should be marked invalid once coast budget is exceeded"
