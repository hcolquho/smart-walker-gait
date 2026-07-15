"""
Real 2D pose backend using MMPose (RTMPose for live/low-latency mode,
ViTPose for offline/high-fidelity mode -- both live in the same MMPose API,
selected purely by which config/checkpoint you load).

NOT wired to a specific checkpoint here -- fill in `config_path` /
`checkpoint_path` once you've picked model sizes for your target hardware.

Install (heavy, do this only when ready to move off DummyPoseEstimator2D):
    pip install -U openmim
    mim install mmengine "mmcv>=2.0.1" mmpose mmdet

Suggested starting checkpoints (both COCO-keypoint compatible, 13-joint
subset covered by our JOINT_NAMES is a subset of COCO-17):
    RTMPose (live):  rtmpose-m_8xb256-420e_coco-256x192
    ViTPose  (offline/high-fidelity): vitpose-b_8xb64-210e_coco-256x192
"""
from __future__ import annotations
import numpy as np

from walker_gait.core.types import Skeleton2D, TrackedDetection, NUM_JOINTS, JOINT_INDEX, JOINT_NAMES
from walker_gait.pose2d.base import PoseEstimator2D, POSE2D_REGISTRY

# COCO-17 keypoint name -> index, for mapping model output onto our joint set.
_COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Our set adds heel/toe, which COCO-17 doesn't have -- those come from a
# foot-keypoint model (e.g. RTMPose-wholebody) or are left as low-confidence
# extrapolations from ankle until you add a foot-keypoint checkpoint.


def _register_backend(key: str):
    @POSE2D_REGISTRY.register(key)
    class _MMPoseEstimator(PoseEstimator2D):
        def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
            from mmpose.apis import init_model  # lazy import, heavy dependency
            self.model = init_model(config_path, checkpoint_path, device=device)

        def estimate(self, rgb: np.ndarray, detection: TrackedDetection, timestamp: float) -> Skeleton2D:
            from mmpose.apis import inference_topdown
            bbox = np.array([detection.bbox], dtype=np.float32)
            results = inference_topdown(self.model, rgb, bboxes=bbox, bbox_format="xyxy")

            keypoints = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            scores = np.zeros((NUM_JOINTS,), dtype=np.float32)
            if results:
                pred = results[0].pred_instances
                coco_kpts = pred.keypoints[0]        # (17, 2)
                coco_scores = pred.keypoint_scores[0]  # (17,)
                for coco_idx, name in enumerate(_COCO_NAMES):
                    if name in JOINT_INDEX:
                        j = JOINT_INDEX[name]
                        keypoints[j] = coco_kpts[coco_idx]
                        scores[j] = coco_scores[coco_idx]
                # Heel/toe: extrapolate along the shin direction from ankle
                # until a wholebody/foot checkpoint is wired in.
                for side in ("left", "right"):
                    ankle = keypoints[JOINT_INDEX[f"{side}_ankle"]]
                    knee = keypoints[JOINT_INDEX[f"{side}_knee"]]
                    shin_dir = ankle - knee
                    keypoints[JOINT_INDEX[f"{side}_heel"]] = ankle + 0.15 * shin_dir
                    keypoints[JOINT_INDEX[f"{side}_toe"]] = ankle + 0.35 * shin_dir
                    scores[JOINT_INDEX[f"{side}_heel"]] = scores[JOINT_INDEX[f"{side}_ankle"]] * 0.5
                    scores[JOINT_INDEX[f"{side}_toe"]] = scores[JOINT_INDEX[f"{side}_ankle"]] * 0.5

            return Skeleton2D(keypoints=keypoints, scores=scores,
                               track_id=detection.track_id, timestamp=timestamp)

    return _MMPoseEstimator


RTMPoseEstimator = _register_backend("rtmpose")
ViTPoseEstimator = _register_backend("vitpose")
