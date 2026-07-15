"""
MonoDepthProxySource wraps any other FrameSource (typically WebcamSource) and
adds a *simulated* depth channel using a monocular depth model. This is NOT
metrically accurate (monocular depth is scale-ambiguous) but lets you exercise
the entire backprojection / smoothing / gait-metric code path -- shapes,
units, failure handling -- before the Femto Bolt provides real depth.

Requires (install only when you want to use this backend):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers timm

Model used: "Intel/dpt-hybrid-midas" via HuggingFace transformers, lazily
imported so the rest of the package has zero heavy ML dependencies.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from walker_gait.core.types import Frame, CameraIntrinsics
from walker_gait.frame_source.base import FrameSource, FRAME_SOURCE_REGISTRY


@FRAME_SOURCE_REGISTRY.register("mono_depth_proxy")
class MonoDepthProxySource(FrameSource):
    def __init__(self, wrapped: FrameSource, approx_scale_m: float = 2.0,
                 model_name: str = "Intel/dpt-hybrid-midas"):
        self.wrapped = wrapped
        self.approx_scale_m = approx_scale_m
        self._model = None
        self._processor = None
        self._model_name = model_name

    def is_depth_available(self) -> bool:
        return True  # simulated

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        self._torch = torch
        self._processor = DPTImageProcessor.from_pretrained(self._model_name)
        self._model = DPTForDepthEstimation.from_pretrained(self._model_name)
        self._model.eval()

    def get_frame(self) -> Optional[Frame]:
        frame = self.wrapped.get_frame()
        if frame is None:
            return None
        self._lazy_load()
        import PIL.Image as Image
        img = Image.fromarray(frame.rgb)
        inputs = self._processor(images=img, return_tensors="pt")
        with self._torch.no_grad():
            depth_pred = self._model(**inputs).predicted_depth
        depth_pred = self._torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1), size=frame.rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze().numpy()

        # Inverse-depth model output -> rescale into a plausible metric range
        # so downstream code sees "meters", even though absolute scale is not
        # trustworthy. Relative depth ordering between joints is still useful
        # for validating pipeline plumbing.
        d = depth_pred - depth_pred.min()
        d = d / (d.max() + 1e-6)
        depth_m = self.approx_scale_m * (1.0 - d) + 0.5  # closer objects -> larger raw value

        frame.depth = depth_m.astype(np.float32)
        if frame.intrinsics is None:
            h, w = frame.rgb.shape[:2]
            frame.intrinsics = CameraIntrinsics(fx=525.0 * w / 640, fy=525.0 * w / 640,
                                                 cx=w / 2, cy=h / 2, width=w, height=h)
        return frame

    def close(self) -> None:
        self.wrapped.close()
