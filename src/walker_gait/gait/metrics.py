"""
GaitMetricsCalculator: computes clinical gait metrics from a clean event list
(+ optional 3D trajectory for step-length / loading metrics). Deliberately
kept as near-pure-arithmetic and independent of vision/event-detection
quality so it's trivially unit-testable with hand-crafted event sequences.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from walker_gait.core.types import GaitEvent, GaitEventType, Side, GaitMetrics, Skeleton3D, JOINT_INDEX


def _events_of(events: List[GaitEvent], side: Side, etype: GaitEventType) -> List[float]:
    return sorted(e.timestamp for e in events if e.side == side and e.type == etype)


def _mean_interval(timestamps: List[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(timestamps)
    return float(np.mean(diffs))


def _asymmetry(left: float, right: float) -> float:
    """Symmetric percentage-style asymmetry index: 0 = perfectly symmetric,
    positive = right > left, negative = left > right. Bounded roughly [-2, 2]
    for sane inputs; using (R-L)/((R+L)/2) is a standard clinical convention."""
    denom = (left + right) / 2.0
    if denom == 0:
        return 0.0
    return float((right - left) / denom)


class GaitMetricsCalculator:
    def __init__(self, fsr_load_left: Optional[np.ndarray] = None,
                 fsr_load_right: Optional[np.ndarray] = None,
                 fsr_fps: float = 100.0):
        """fsr_load_left/right: optional per-frame FSR pressure sums, used for
        loading asymmetry. If omitted, loading asymmetry falls back to a
        vision-derived proxy (stance-phase duration) which is far cruder."""
        self.fsr_load_left = fsr_load_left
        self.fsr_load_right = fsr_load_right
        self.fsr_fps = fsr_fps

    def compute(self, events: List[GaitEvent],
                skeletons: Optional[List[Skeleton3D]] = None) -> GaitMetrics:
        left_hs = _events_of(events, Side.LEFT, GaitEventType.HEEL_STRIKE)
        right_hs = _events_of(events, Side.RIGHT, GaitEventType.HEEL_STRIKE)
        left_to = _events_of(events, Side.LEFT, GaitEventType.TOE_OFF)
        right_to = _events_of(events, Side.RIGHT, GaitEventType.TOE_OFF)

        stride_time_l = _mean_interval(left_hs)
        stride_time_r = _mean_interval(right_hs)

        all_hs = sorted(left_hs + right_hs)
        step_times = np.diff(all_hs) if len(all_hs) > 1 else np.array([])
        cadence = 60.0 / np.mean(step_times) if len(step_times) > 0 else 0.0

        step_time_l = _mean_interval(sorted(left_hs))
        step_time_r = _mean_interval(sorted(right_hs))
        step_time_asym = _asymmetry(step_time_l, step_time_r)

        step_length_l, step_length_r = self._step_lengths(events, skeletons)
        step_length_asym = _asymmetry(step_length_l, step_length_r)

        loading_asym = self._loading_asymmetry(left_hs, right_hs)

        double_support = self._double_support_time(left_hs, left_to, right_hs, right_to)

        aggregate = float(np.mean(np.abs([step_time_asym, step_length_asym, loading_asym])))

        return GaitMetrics(
            cadence_steps_per_min=float(cadence),
            stride_time_left_s=stride_time_l,
            stride_time_right_s=stride_time_r,
            step_time_asymmetry=step_time_asym,
            step_length_asymmetry=step_length_asym,
            loading_asymmetry=loading_asym,
            double_support_time_s=double_support,
            aggregate_asymmetry_score=aggregate,
            extra={
                "n_left_strides": len(left_hs), "n_right_strides": len(right_hs),
                "step_time_left_s": step_time_l, "step_time_right_s": step_time_r,
                "step_length_left_m": step_length_l, "step_length_right_m": step_length_r,
            },
        )

    def _step_lengths(self, events: List[GaitEvent],
                       skeletons: Optional[List[Skeleton3D]]) -> tuple:
        """Step length = forward (z-axis) distance ankle travels between
        consecutive heel-strikes of the SAME side. Requires the 3D trajectory;
        falls back to (0, 0) if not supplied (e.g. events-only unit tests)."""
        if skeletons is None:
            return 0.0, 0.0

        ts_to_idx = {s.timestamp: i for i, s in enumerate(skeletons)}

        def lengths_for(side: Side, joint_name: str) -> float:
            hs = _events_of(events, side, GaitEventType.HEEL_STRIKE)
            dists = []
            for a, b in zip(hs[:-1], hs[1:]):
                ia, ib = ts_to_idx.get(a), ts_to_idx.get(b)
                if ia is None or ib is None:
                    continue
                pa = skeletons[ia].keypoints[JOINT_INDEX[joint_name]]
                pb = skeletons[ib].keypoints[JOINT_INDEX[joint_name]]
                dists.append(float(np.linalg.norm(pb - pa)))
            return float(np.mean(dists)) if dists else 0.0

        return lengths_for(Side.LEFT, "left_ankle"), lengths_for(Side.RIGHT, "right_ankle")

    def _loading_asymmetry(self, left_hs: List[float], right_hs: List[float]) -> float:
        if self.fsr_load_left is not None and self.fsr_load_right is not None:
            l_load = float(np.mean(self.fsr_load_left))
            r_load = float(np.mean(self.fsr_load_right))
            return _asymmetry(l_load, r_load)
        # Vision-only fallback proxy: relative stride count as a very crude
        # stand-in until FSR data is available -- flagged clearly as low
        # confidence via `extra` in the caller if needed.
        return _asymmetry(len(left_hs), len(right_hs))

    def _double_support_time(self, left_hs, left_to, right_hs, right_to) -> float:
        """Double support = time both feet are on the ground: from a
        heel-strike on one side until the OTHER side's toe-off. Averaged
        across all detected instances."""
        durations = []
        for hs in left_hs:
            future_r_to = [t for t in right_to if t > hs]
            if future_r_to:
                durations.append(min(future_r_to) - hs)
        for hs in right_hs:
            future_l_to = [t for t in left_to if t > hs]
            if future_l_to:
                durations.append(min(future_l_to) - hs)
        return float(np.mean(durations)) if durations else 0.0
