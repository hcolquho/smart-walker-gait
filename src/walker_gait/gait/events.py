"""
Gait event detection: converts a stream of smoothed 3D skeletons into discrete
heel-strike / toe-off events per side.

Kept deliberately separate from GaitMetricsCalculator (events.py vs
metrics.py): event detection is the noisy, algorithm-sensitive part (depends
on joint trajectories, thresholds, sensor quality); metric computation from a
clean event list is close to pure arithmetic. Splitting them means you can
improve/replace detection later without touching metric formulas, and you can
unit test metrics with hand-crafted event sequences independent of vision
quality.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np

from walker_gait.core.types import Skeleton3D, GaitEvent, GaitEventType, Side, JOINT_INDEX


class VerticalVelocityEventDetector:
    """Heuristic: heel-strike ~ local minimum of heel height with near-zero
    downward velocity crossing; toe-off ~ local minimum of toe height with
    upward velocity onset. This is a simple, dependency-free baseline --
    swap for a more sophisticated detector (e.g. Zeni et al. coordinate-based
    method, or FSR-ground-truth-trained classifier) once real data exists.
    """

    def __init__(self, fps: float = 30.0, min_event_spacing_s: float = 0.25,
                 pre_smooth_window: int = 5):
        self.dt = 1.0 / fps
        self.min_spacing_frames = int(min_event_spacing_s * fps)
        self.pre_smooth_window = pre_smooth_window  # odd, moving-average width
        self._history = {Side.LEFT: [], Side.RIGHT: []}
        self._last_event_frame = {(Side.LEFT, GaitEventType.HEEL_STRIKE): -999,
                                   (Side.LEFT, GaitEventType.TOE_OFF): -999,
                                   (Side.RIGHT, GaitEventType.HEEL_STRIKE): -999,
                                   (Side.RIGHT, GaitEventType.TOE_OFF): -999}
        self._frame_idx = 0

    def reset(self):
        self._history = {Side.LEFT: [], Side.RIGHT: []}
        self._frame_idx = 0

    def process_sequence(self, skeletons: List[Skeleton3D],
                          fsr_left: Optional[np.ndarray] = None,
                          fsr_right: Optional[np.ndarray] = None) -> List[GaitEvent]:
        """Batch mode: run over an entire buffered sequence (simplest and most
        robust way to run this -- local minima detection needs some lookahead).

        fsr_left / fsr_right (optional): per-frame pressure arrays from the
        Interlink FSRs. When provided, heel-strike is corroborated/replaced by
        the FSR rising edge (FSR ground contact is a cleaner signal than
        vision-derived heel height) -- same cross-validation pattern used for
        the IMU.
        """
        events: List[GaitEvent] = []
        for side, heel_name, toe_name in [
            (Side.LEFT, "left_heel", "left_toe"),
            (Side.RIGHT, "right_heel", "right_toe"),
        ]:
            heel_y = np.array([s.keypoints[JOINT_INDEX[heel_name], 1] for s in skeletons])
            toe_y = np.array([s.keypoints[JOINT_INDEX[toe_name], 1] for s in skeletons])
            valid = np.array([s.valid[JOINT_INDEX[heel_name]] for s in skeletons])

            heel_y_smooth = self._moving_average(heel_y)
            toe_y_smooth = self._moving_average(toe_y)

            heel_strikes = self._local_minima(heel_y_smooth, valid, self.min_spacing_frames)
            toe_offs = self._local_minima(toe_y_smooth, valid, self.min_spacing_frames)

            fsr = fsr_left if side == Side.LEFT else fsr_right
            if fsr is not None:
                heel_strikes = self._corroborate_with_fsr(heel_strikes, fsr, skeletons)

            for idx in heel_strikes:
                events.append(GaitEvent(type=GaitEventType.HEEL_STRIKE, side=side,
                                          timestamp=skeletons[idx].timestamp, frame_id=idx))
            for idx in toe_offs:
                events.append(GaitEvent(type=GaitEventType.TOE_OFF, side=side,
                                          timestamp=skeletons[idx].timestamp, frame_id=idx))

        events.sort(key=lambda e: e.timestamp)
        return events

    def _moving_average(self, y: np.ndarray) -> np.ndarray:
        """Light causal-symmetric smoothing to suppress sensor-noise wiggles
        near the shallow, flat bottom of the vertical gait trace, which
        otherwise get double-counted as separate local minima."""
        w = self.pre_smooth_window
        if w <= 1 or len(y) < w:
            return y
        kernel = np.ones(w) / w
        # 'same' mode keeps array length; edges are slightly under-smoothed,
        # which is fine since events are unlikely right at the sequence edge.
        return np.convolve(y, kernel, mode="same")

    def _local_minima(self, y: np.ndarray, valid: np.ndarray,
                       min_spacing_frames: int = 8) -> List[int]:
        """Find indices of local minima (joint at its lowest point = ground
        contact for heel-strike, or push-off point for toe-off), enforcing a
        minimum spacing so sensor jitter doesn't produce spurious events."""
        idxs = []
        n = len(y)
        for i in range(1, n - 1):
            if not (valid[i - 1] and valid[i] and valid[i + 1]):
                continue
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
                if not idxs or (i - idxs[-1]) >= min_spacing_frames:
                    idxs.append(i)
        return idxs

    def _corroborate_with_fsr(self, vision_events: List[int], fsr: np.ndarray,
                               skeletons: List[Skeleton3D],
                               fsr_threshold: float = 0.1,
                               search_window_frames: int = 6) -> List[int]:
        """Snap each vision-derived heel-strike to the nearest FSR rising
        edge within a small search window, since FSR ground-contact timing is
        more reliable than vision-derived heel height."""
        rising_edges = [i for i in range(1, len(fsr))
                         if fsr[i - 1] < fsr_threshold <= fsr[i]]
        if not rising_edges:
            return vision_events

        corroborated = []
        for v_idx in vision_events:
            candidates = [r for r in rising_edges if abs(r - v_idx) <= search_window_frames]
            corroborated.append(min(candidates, key=lambda r: abs(r - v_idx)) if candidates else v_idx)
        return corroborated
