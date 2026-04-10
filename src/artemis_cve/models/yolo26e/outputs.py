from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.utils import ModelOutput


@dataclass
class YOLOEOutput(ModelOutput):
    boxes: list[torch.Tensor] | None = None
    scores: list[torch.Tensor] | None = None
    labels: list[torch.Tensor] | None = None
    masks: list[torch.Tensor | None] | None = None
    raw_results: Any | None = None
