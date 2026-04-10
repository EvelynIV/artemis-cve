from __future__ import annotations

from dataclasses import replace

import numpy as np

from artemis_cve.inferencers.yolo.inferencer import BoxDetection


def _bbox_iou(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]) -> float:
    x1 = max(lhs[0], rhs[0])
    y1 = max(lhs[1], rhs[1])
    x2 = min(lhs[2], rhs[2])
    y2 = min(lhs[3], rhs[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = max(0.0, rhs[2] - rhs[0]) * max(0.0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class BoxDetectionSmoother:
    def __init__(self, alpha: float = 0.35, match_iou_threshold: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], received {alpha}")
        if not 0.0 <= match_iou_threshold <= 1.0:
            raise ValueError(
                "match_iou_threshold must be in [0, 1], "
                f"received {match_iou_threshold}"
            )

        self.alpha = float(alpha)
        self.match_iou_threshold = float(match_iou_threshold)
        self._previous: list[BoxDetection] = []

    def reset(self) -> None:
        self._previous = []

    def smooth(self, detections: list[BoxDetection]) -> list[BoxDetection]:
        if not detections:
            self.reset()
            return []

        matched_previous: set[int] = set()
        smoothed: list[BoxDetection] = []

        for detection in detections:
            best_idx = None
            best_iou = 0.0

            for idx, previous in enumerate(self._previous):
                if idx in matched_previous or previous.class_id != detection.class_id:
                    continue

                iou = _bbox_iou(previous.pixel_xyxy, detection.pixel_xyxy)
                if iou >= self.match_iou_threshold and iou > best_iou:
                    best_idx = idx
                    best_iou = iou

            if best_idx is None:
                smoothed.append(detection)
                continue

            matched_previous.add(best_idx)
            previous = self._previous[best_idx]
            prev_xyxy = np.asarray(previous.pixel_xyxy, dtype=np.float32)
            curr_xyxy = np.asarray(detection.pixel_xyxy, dtype=np.float32)
            next_xyxy = self.alpha * curr_xyxy + (1.0 - self.alpha) * prev_xyxy
            next_score = self.alpha * detection.score + (1.0 - self.alpha) * previous.score

            smoothed.append(
                replace(
                    detection,
                    score=float(next_score),
                    pixel_xyxy=tuple(float(value) for value in next_xyxy.tolist()),
                    normalized_xyxy=detection.normalize(
                        tuple(float(value) for value in next_xyxy.tolist()),
                        detection.image_size,
                    ),
                )
            )

        self._previous = smoothed
        return smoothed
