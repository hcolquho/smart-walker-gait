"""
MotionBERT: the OFFLINE / high-fidelity smoothing tier. Runs over a buffered
window of frames (not streaming/causal like Kalman) -- intended for the
post-session "clinical report" pass, not the live UI, so its added latency
(needing both past AND future frames in the window) is not a problem.

Not wired to a checkpoint yet. Reference implementation to adapt once you're
ready: https://github.com/Walter0807/MotionBERT

Install (heavy — GPU strongly recommended):
    pip install torch torchvision
    git clone https://github.com/Walter0807/MotionBERT
    # download pretrained checkpoint per their README (e.g. FT_MB_release_MB_ft_h36m)

Usage pattern in this codebase: buffer N frames of Skeleton3D (already
Kalman-smoothed or raw), run this once per completed session/segment, use its
output for GaitMetrics computation in the clinical report rather than the
live display.
"""
from __future__ import annotations
from typing import List
import numpy as np

from walker_gait.core.types import Skeleton3D, NUM_JOINTS
from walker_gait.smoothing.base import SMOOTHER_REGISTRY


@SMOOTHER_REGISTRY.register("motionbert")
class MotionBertSmoother:
    """Batch/offline API -- deliberately NOT the same streaming `update()`
    interface as KalmanSkeletonSmoother, because MotionBERT is non-causal by
    design. Call `smooth_sequence()` once you have a full buffered clip.
    """

    def __init__(self, checkpoint_path: str, window_frames: int = 243, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.window_frames = window_frames
        self.device = device
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        raise NotImplementedError(
            "Wire up MotionBERT here once cloned/installed: load architecture, "
            "load `self.checkpoint_path` state dict, move to `self.device`, eval()."
        )
        # Sketch of intended body once the dependency is available:
        #   import torch
        #   from lib.model.DSTformer import DSTformer
        #   self._model = DSTformer(...)
        #   ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        #   self._model.load_state_dict(ckpt['model_pos'])
        #   self._model.to(self.device).eval()

    def smooth_sequence(self, frames: List[Skeleton3D]) -> List[Skeleton3D]:
        """Input/output: list of Skeleton3D across a full session/segment.
        Handles occlusion by filling gaps using attention across the window
        (frames before + after a gap inform the missing joint's likely path)."""
        self._lazy_load()
        raise NotImplementedError("Fill in forward pass + windowed inference once model is loaded.")
