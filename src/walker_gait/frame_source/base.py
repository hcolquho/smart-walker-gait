from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from walker_gait.core.types import Frame
from walker_gait.core.factory import Registry

FRAME_SOURCE_REGISTRY = Registry("frame_source")


class FrameSource(ABC):
    """Every sensor backend (webcam, video file, synthetic rig, Femto Bolt...)
    implements this. Nothing downstream should ever import a concrete backend
    directly -- always go through FRAME_SOURCE_REGISTRY.build(name, **cfg).
    """

    @abstractmethod
    def get_frame(self) -> Optional[Frame]:
        """Return the next Frame, or None when the stream is exhausted."""
        raise NotImplementedError

    @abstractmethod
    def is_depth_available(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                return
            yield frame
