from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoModel
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes


@dataclass(frozen=True, slots=True)
class BoxDetection:
    class_id: int
    class_name: str
    score: float
    pixel_xyxy: tuple[float, float, float, float]
    normalized_xyxy: tuple[float, float, float, float]
    image_size: tuple[int, int]

    @staticmethod
    def normalize(
        pixel_xyxy: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        height, width = image_size
        if height <= 0 or width <= 0:
            return 0.0, 0.0, 0.0, 0.0

        x1, y1, x2, y2 = pixel_xyxy
        return (
            float(np.clip(x1 / width, 0.0, 1.0)),
            float(np.clip(y1 / height, 0.0, 1.0)),
            float(np.clip(x2 / width, 0.0, 1.0)),
            float(np.clip(y2 / height, 0.0, 1.0)),
        )


class YoloBoxInferencer:
    DEFAULT_IMGSZ = 640

    def __init__(
        self,
        model_dir: str | Path,
        class_names: Sequence[str] | None = None,
        device: str | torch.device = "cpu",
        imgsz: int | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        default_class_names = tuple(str(name) for name in getattr(self.config, "default_classes", []) if str(name))
        provided_class_names = tuple(str(name).strip() for name in (class_names or []) if str(name).strip())
        self.class_names = provided_class_names or default_class_names
        if not self.class_names and bool(getattr(self.config, "open_vocab", False)):
            raise ValueError(
                "class_names must be provided for this model. "
                "Pass them via CLI/env or add default_classes to the model config."
            )

        self.imgsz = int(imgsz or getattr(self.config, "image_size", self.DEFAULT_IMGSZ))
        strides = getattr(self.config, "stride", None) or [32]
        self.stride = int(max(int(value) for value in strides))

        self._letterbox = LetterBox(
            new_shape=(self.imgsz, self.imgsz),
            auto=False,
            scale_fill=False,
            scaleup=True,
            stride=self.stride,
        )

    def _resolve_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]

        id2label = getattr(self.config, "id2label", {}) or {}
        if class_id in id2label:
            return str(id2label[class_id])
        if str(class_id) in id2label:
            return str(id2label[str(class_id)])
        return f"class_{class_id}"

    def _preprocess(
        self,
        bgr: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                "Expected a BGR image with shape (H, W, 3), "
                f"received {tuple(bgr.shape)}"
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed = self._letterbox(image=rgb)
        tensor = (
            torch.from_numpy(processed)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(self.device)
        return tensor, tuple(int(v) for v in rgb.shape[:2]), tuple(int(v) for v in processed.shape[:2])

    def _convert_outputs(
        self,
        outputs: Any,
        original_shape: tuple[int, int],
        processed_shape: tuple[int, int],
        score_threshold: float,
        max_detections: int | None,
    ) -> list[BoxDetection]:
        boxes_batch = outputs.boxes or []
        scores_batch = outputs.scores or []
        labels_batch = outputs.labels or []
        if not boxes_batch:
            return []

        boxes = boxes_batch[0]
        scores = scores_batch[0]
        labels = labels_batch[0]
        if boxes.numel() == 0:
            return []

        scaled_boxes = scale_boxes(processed_shape, boxes.clone(), original_shape)
        height, width = original_shape
        detections: list[BoxDetection] = []

        for box_tensor, score_tensor, label_tensor in zip(scaled_boxes, scores, labels):
            score = float(score_tensor.item())
            if score < score_threshold:
                continue

            x1, y1, x2, y2 = [float(value) for value in box_tensor.tolist()]
            pixel_xyxy = (
                float(np.clip(x1, 0.0, width)),
                float(np.clip(y1, 0.0, height)),
                float(np.clip(x2, 0.0, width)),
                float(np.clip(y2, 0.0, height)),
            )
            class_id = int(label_tensor.item())
            detections.append(
                BoxDetection(
                    class_id=class_id,
                    class_name=self._resolve_class_name(class_id),
                    score=score,
                    pixel_xyxy=pixel_xyxy,
                    normalized_xyxy=BoxDetection.normalize(pixel_xyxy, original_shape),
                    image_size=original_shape,
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        if max_detections is not None and max_detections > 0:
            detections = detections[:max_detections]
        return detections

    @torch.no_grad()
    def infer(
        self,
        bgr: np.ndarray,
        score_threshold: float = 0.0,
        max_detections: int | None = None,
    ) -> list[BoxDetection]:
        tensor, original_shape, processed_shape = self._preprocess(bgr)
        request_kwargs: dict[str, Any] = {
            "pixel_values": tensor,
            "class_names": list(self.class_names),
        }
        if max_detections is not None and max_detections > 0:
            request_kwargs["max_det"] = int(max_detections)

        outputs = self.model(**request_kwargs)
        return self._convert_outputs(
            outputs=outputs,
            original_shape=original_shape,
            processed_shape=processed_shape,
            score_threshold=float(score_threshold),
            max_detections=max_detections,
        )

    def infer_batch(
        self,
        images: Iterable[np.ndarray],
        score_threshold: float = 0.0,
        max_detections: int | None = None,
    ) -> list[list[BoxDetection]]:
        return [
            self.infer(
                bgr=image,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            for image in images
        ]
