import numpy as np
from walker_gait.core.types import Skeleton2D, NUM_JOINTS
from walker_gait.depth.backprojection import DepthBackprojector
from walker_gait.frame_source.synthetic import SyntheticSource


def test_backprojection_recovers_ground_truth_within_tolerance():
    src = SyntheticSource(fps=30.0, duration_s=0.5, noise_std_px=0.0)
    frame = src.get_frame()

    skeleton2d = Skeleton2D(keypoints=frame.gt_2d, scores=np.ones(NUM_JOINTS))
    bp = DepthBackprojector(window=5)
    skeleton3d = bp.backproject_skeleton(skeleton2d, frame.depth, frame.intrinsics)

    assert skeleton3d.valid.all(), "all joints should backproject successfully on clean synthetic data"
    err = np.linalg.norm(skeleton3d.keypoints - frame.gt_3d, axis=1)
    # Loose tolerance: ankle/heel/toe are only a few pixels apart in this
    # synthetic rig, so their depth-paint windows legitimately overlap.
    assert (err < 0.15).all(), f"backprojection error too high: max {err.max():.4f} m"


def test_missing_depth_marks_joint_invalid_not_garbage():
    src = SyntheticSource(fps=30.0, duration_s=0.5, noise_std_px=0.0)
    frame = src.get_frame()
    depth_with_hole = frame.depth.copy()
    depth_with_hole[:, :] = 0.0  # simulate total depth dropout

    skeleton2d = Skeleton2D(keypoints=frame.gt_2d, scores=np.ones(NUM_JOINTS))
    bp = DepthBackprojector(window=5)
    skeleton3d = bp.backproject_skeleton(skeleton2d, depth_with_hole, frame.intrinsics)

    assert not skeleton3d.valid.any(), "with zero valid depth, every joint must be marked invalid"


def test_median_window_reduces_impact_of_single_pixel_noise_spike():
    src = SyntheticSource(fps=30.0, duration_s=0.5, noise_std_px=0.0)
    frame = src.get_frame()
    noisy_depth = frame.depth.copy()

    # inject an extreme single-pixel spike right at one joint's exact pixel
    u, v = int(round(frame.gt_2d[7, 0])), int(round(frame.gt_2d[7, 1]))
    noisy_depth[v, u] = 100.0  # wildly wrong reading

    skeleton2d = Skeleton2D(keypoints=frame.gt_2d, scores=np.ones(NUM_JOINTS))
    bp = DepthBackprojector(window=5, max_depth_m=6.0)
    skeleton3d = bp.backproject_skeleton(skeleton2d, noisy_depth, frame.intrinsics)

    # the spike (100m) is above max_depth_m so it's excluded from the median;
    # the joint should still resolve correctly from the surrounding window
    assert skeleton3d.valid[7]
    assert abs(skeleton3d.keypoints[7, 2] - frame.gt_3d[7, 2]) < 0.15
