from walker_gait.core.types import Detection
from walker_gait.tracking.iou_tracker import IouTracker


def test_track_id_persists_across_frames_with_overlap():
    tracker = IouTracker(iou_threshold=0.3, max_misses=5)
    det1 = Detection(bbox=(100, 100, 200, 300), confidence=0.9)
    out1 = tracker.update([det1], timestamp=0.0)
    assert len(out1) == 1
    tid = out1[0].track_id

    det2 = Detection(bbox=(105, 102, 205, 302), confidence=0.9)  # small shift, overlaps
    out2 = tracker.update([det2], timestamp=0.033)
    assert out2[0].track_id == tid


def test_track_survives_brief_occlusion_gap():
    tracker = IouTracker(iou_threshold=0.3, max_misses=5)
    det = Detection(bbox=(100, 100, 200, 300), confidence=0.9)
    out1 = tracker.update([det], timestamp=0.0)
    tid = out1[0].track_id

    # 3 frames with NO detection (occlusion) -- within max_misses=5
    tracker.update([], timestamp=0.033)
    tracker.update([], timestamp=0.066)
    tracker.update([], timestamp=0.099)

    det_again = Detection(bbox=(102, 101, 202, 301), confidence=0.9)
    out_final = tracker.update([det_again], timestamp=0.132)
    assert out_final[0].track_id == tid, "track_id should survive a brief occlusion"


def test_track_dropped_after_max_misses_exceeded():
    tracker = IouTracker(iou_threshold=0.3, max_misses=2)
    det = Detection(bbox=(100, 100, 200, 300), confidence=0.9)
    out1 = tracker.update([det], timestamp=0.0)
    tid = out1[0].track_id

    for i in range(1, 5):  # exceeds max_misses=2
        tracker.update([], timestamp=i * 0.033)

    det_again = Detection(bbox=(102, 101, 202, 301), confidence=0.9)
    out_final = tracker.update([det_again], timestamp=0.2)
    assert out_final[0].track_id != tid, "track should have been dropped and reassigned a new id"
