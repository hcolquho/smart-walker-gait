import numpy as np
from walker_gait.core.types import Skeleton3D, Side, GaitEventType, JOINT_INDEX, NUM_JOINTS
from walker_gait.gait.events import VerticalVelocityEventDetector
from walker_gait.frame_source.synthetic import SyntheticSource


def _skeletons_from_synthetic(duration_s=6.0, cadence=100.0):
    src = SyntheticSource(fps=30.0, duration_s=duration_s, cadence_steps_per_min=cadence, noise_std_px=0.0)
    skeletons = []
    for frame in src:
        pts = frame.gt_3d
        valid = np.ones(NUM_JOINTS, dtype=bool)
        skeletons.append(Skeleton3D(keypoints=pts, valid=valid, timestamp=frame.timestamp))
    return skeletons


def test_event_count_roughly_matches_expected_cadence():
    cadence = 100.0  # steps/min combined across both legs -> ~50 steps/min per leg
    duration_s = 6.0
    skeletons = _skeletons_from_synthetic(duration_s=duration_s, cadence=cadence)

    detector = VerticalVelocityEventDetector(fps=30.0)
    events = detector.process_sequence(skeletons)

    heel_strikes = [e for e in events if e.type == GaitEventType.HEEL_STRIKE]
    expected_total_steps = (cadence / 60.0) * duration_s  # combined both legs
    # Allow generous tolerance -- this is a heuristic local-minima detector,
    # not a claim of frame-perfect event detection.
    assert abs(len(heel_strikes) - expected_total_steps) <= max(3, 0.3 * expected_total_steps)


def test_fsr_corroboration_snaps_vision_event_to_fsr_edge():
    skeletons = _skeletons_from_synthetic(duration_s=4.0, cadence=100.0)
    n = len(skeletons)

    detector = VerticalVelocityEventDetector(fps=30.0)
    baseline_events = detector.process_sequence(skeletons)
    baseline_left_hs = [e.frame_id for e in baseline_events
                         if e.type == GaitEventType.HEEL_STRIKE and e.side == Side.LEFT]
    assert baseline_left_hs, "expected at least one left heel-strike from vision alone"

    # Simulate an FSR rising edge offset by a few frames from the true vision
    # event, and confirm corroboration pulls the reported event onto the FSR
    # edge (FSR ground-contact timing is treated as more trustworthy).
    true_frame = baseline_left_hs[0]
    fsr_edge_frame = min(true_frame + 3, n - 2)
    fsr_left = np.zeros(n)
    fsr_left[fsr_edge_frame:] = 1.0

    events = detector.process_sequence(skeletons, fsr_left=fsr_left, fsr_right=None)
    left_hs_frames = [e.frame_id for e in events
                       if e.type == GaitEventType.HEEL_STRIKE and e.side == Side.LEFT]

    assert fsr_edge_frame in left_hs_frames, (
        f"expected corroborated event to snap exactly onto FSR edge frame {fsr_edge_frame}, "
        f"got {left_hs_frames}"
    )
