import numpy as np
from walker_gait.frame_source.synthetic import SyntheticSource


def test_synthetic_source_produces_frames_until_exhausted():
    src = SyntheticSource(fps=30.0, duration_s=1.0)
    frames = list(src)
    assert len(frames) == 30
    assert src.get_frame() is None  # exhausted


def test_ground_truth_2d_matches_projection_of_3d():
    """The reported gt_2d must be the exact pinhole projection of gt_3d --
    if this test fails, the synthetic harness itself is broken and nothing
    downstream can be trusted."""
    src = SyntheticSource(fps=30.0, duration_s=0.5, noise_std_px=0.0)
    frame = src.get_frame()
    reprojected = src.project(frame.gt_3d)
    assert np.allclose(reprojected, frame.gt_2d, atol=1e-3)


def test_depth_channel_matches_ground_truth_z_at_joint_pixels():
    src = SyntheticSource(fps=30.0, duration_s=0.5, noise_std_px=0.0)
    frame = src.get_frame()
    for j in range(frame.gt_3d.shape[0]):
        u, v = int(round(frame.gt_2d[j, 0])), int(round(frame.gt_2d[j, 1]))
        if 0 <= u < frame.depth.shape[1] and 0 <= v < frame.depth.shape[0]:
            # Loose tolerance: ankle/heel/toe are only a few pixels apart, so
            # their depth-paint windows legitimately overlap and the last
            # joint painted wins at shared pixels (expected, not a bug).
            assert abs(frame.depth[v, u] - frame.gt_3d[j, 2]) < 0.15


def test_injected_asymmetry_changes_leg_phase_difference():
    """Sanity check that left_right_asymmetry actually perturbs the gait
    model -- important since several downstream tests validate metrics
    AGAINST a known injected asymmetry value."""
    sym = SyntheticSource(fps=30.0, duration_s=2.0, left_right_asymmetry=0.0)
    asym = SyntheticSource(fps=30.0, duration_s=2.0, left_right_asymmetry=0.4)
    sym_traj = np.stack([sym.ground_truth_3d(i)[7] for i in range(60)])
    asym_traj = np.stack([asym.ground_truth_3d(i)[7] for i in range(60)])
    assert not np.allclose(sym_traj, asym_traj)
