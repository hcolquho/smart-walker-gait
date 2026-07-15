"""
Lightweight IOU-based tracker (SORT-lite, no Kalman motion model needed here
since the detector already runs every frame and the walker is single-person).

Design choice: a full DeepSORT-style appearance-embedding tracker is overkill
-- we only need to re-identify "the same person" after a few missed/occluded
frames, not distinguish between multiple people. IOU + a short grace period
covers this at near-zero compute cost.
"""
from __future__ import annotations
from typing import List, Optional

from walker_gait.core.types import Detection, TrackedDetection
from walker_gait.tracking.base import Tracker, TRACKER_REGISTRY


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("track_id", "bbox", "last_seen_ts", "misses")

    def __init__(self, track_id: int, bbox: tuple, ts: float):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen_ts = ts
        self.misses = 0


@TRACKER_REGISTRY.register("iou")
class IouTracker(Tracker):
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: List[_Track] = []
        self._next_id = 0

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0

    def update(self, detections: List[Detection], timestamp: float) -> List[TrackedDetection]:
        results: List[TrackedDetection] = []
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy best-IOU matching (fine for single-person / few-detection case)
        for ti, track in enumerate(self._tracks):
            best_iou, best_di = 0.0, None
            for di in unmatched_dets:
                iou = _iou(track.bbox, detections[di].bbox)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_di is not None and best_iou >= self.iou_threshold:
                det = detections[best_di]
                track.bbox = det.bbox
                track.last_seen_ts = timestamp
                track.misses = 0
                results.append(TrackedDetection(bbox=det.bbox, confidence=det.confidence,
                                                  class_id=det.class_id, track_id=track.track_id))
                unmatched_dets.remove(best_di)
                matched_track_ids.add(track.track_id)

        # Age out unmatched tracks; drop after max_misses consecutive frames
        alive_tracks = []
        for track in self._tracks:
            if track.track_id in matched_track_ids:
                alive_tracks.append(track)
            else:
                track.misses += 1
                if track.misses <= self.max_misses:
                    alive_tracks.append(track)
        self._tracks = alive_tracks

        # New tracks for leftover detections
        for di in unmatched_dets:
            det = detections[di]
            new_track = _Track(self._next_id, det.bbox, timestamp)
            self._tracks.append(new_track)
            results.append(TrackedDetection(bbox=det.bbox, confidence=det.confidence,
                                              class_id=det.class_id, track_id=new_track.track_id))
            self._next_id += 1

        return results
