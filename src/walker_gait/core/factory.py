"""
Generic registry/factory so every stage (frame source, tracker, pose
estimator, smoother, ...) can be built from a YAML config string without
if/elif ladders scattered through the codebase.

Usage:
    REGISTRY = Registry("pose_estimator")

    @REGISTRY.register("rtmpose")
    class RTMPoseEstimator(PoseEstimator2D):
        ...

    estimator = REGISTRY.build(cfg["backend"], **cfg.get("params", {}))
"""
from __future__ import annotations
from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Type] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        def _wrap(cls: Type[T]) -> Type[T]:
            if key in self._entries:
                raise ValueError(f"[{self.name}] backend '{key}' already registered")
            self._entries[key] = cls
            return cls
        return _wrap

    def build(self, key: str, **kwargs):
        if key not in self._entries:
            raise KeyError(
                f"[{self.name}] unknown backend '{key}'. "
                f"Available: {sorted(self._entries.keys())}"
            )
        return self._entries[key](**kwargs)

    def available(self):
        return sorted(self._entries.keys())
