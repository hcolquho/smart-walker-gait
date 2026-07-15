from __future__ import annotations
from typing import Optional

from walker_gait.core.types import Frame, CameraIntrinsics
from walker_gait.frame_source.base import FrameSource, FRAME_SOURCE_REGISTRY


@FRAME_SOURCE_REGISTRY.register("video_file")
class VideoFileSource(FrameSource):
    """Reads pre-recorded gait videos for repeatable regression testing.
    RGB only unless a matching depth video/npz sidecar is provided later."""

    def __init__(self, path: str, intrinsics: Optional[dict] = None):
        import cv2
        self._cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.intrinsics = CameraIntrinsics.from_dict(intrinsics) if intrinsics else None
        self._frame_id = 0

    def is_depth_available(self) -> bool:
        return False

    def get_frame(self) -> Optional[Frame]:
        ok, bgr = self.cap.read()
        if not ok:
            return None
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        ts = self._frame_id / self.fps
        frame = Frame(rgb=rgb, depth=None, timestamp=ts,
                       intrinsics=self.intrinsics, frame_id=self._frame_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self.cap.release()
