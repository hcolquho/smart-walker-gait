from __future__ import annotations
import time
from typing import Optional

from walker_gait.core.types import Frame, CameraIntrinsics
from walker_gait.frame_source.base import FrameSource, FRAME_SOURCE_REGISTRY


@FRAME_SOURCE_REGISTRY.register("webcam")
class WebcamSource(FrameSource):
    """RGB-only source (no depth) for any USB/laptop webcam. Main dev driver
    for the detector / 2D pose / tracker stages while waiting on the Bolt."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720,
                 intrinsics: Optional[dict] = None):
        import cv2  # local import: keep cv2 optional for non-camera dev machines
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")
        self.intrinsics = CameraIntrinsics.from_dict(intrinsics) if intrinsics else None
        self._frame_id = 0

    def is_depth_available(self) -> bool:
        return False

    def get_frame(self) -> Optional[Frame]:
        ok, bgr = self.cap.read()
        if not ok:
            return None
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        frame = Frame(rgb=rgb, depth=None, timestamp=time.time(),
                       intrinsics=self.intrinsics, frame_id=self._frame_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self.cap.release()
