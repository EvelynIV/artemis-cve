from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .inferencer import BoxDetection, YoloBoxInferencer


class SharedYoloBoxInferencer(YoloBoxInferencer):
    def __init__(
        self,
        model_dir: str | Path,
        class_names: Sequence[str] | None = None,
        device: str | torch.device = "cpu",
        imgsz: int | None = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            class_names=class_names,
            device=device,
            imgsz=imgsz,
        )
        self._lock = threading.Lock()

    def infer(
        self,
        bgr: np.ndarray,
        score_threshold: float = 0.0,
        max_detections: int | None = None,
    ) -> list[BoxDetection]:
        with self._lock:
            return super().infer(
                bgr=bgr,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
