from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel

from .backend import YOLOEBackend
from .configuration import YOLOEConfig
from .outputs import YOLOEOutput


class YOLOEModel(PreTrainedModel):
    config_class = YOLOEConfig
    base_model_prefix = "yoloe"
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_missing = [r".*"]
    _keys_to_ignore_on_load_unexpected = [r".*"]

    def __init__(self, config: YOLOEConfig) -> None:
        super().__init__(config)
        self.runtime_backend = YOLOEBackend(config)
        self.post_init()

    def _get_backend(self) -> Any:
        return self.runtime_backend.get()

    def _prepare_class_names(
        self,
        class_names: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if class_names is None:
            names = list(self.config.default_classes)
            if not names and self.config.open_vocab:
                raise ValueError("class_names must be provided for open-vocabulary inference.")
            return names

        names = list(class_names)
        if not names and self.config.open_vocab:
            raise ValueError("class_names must be provided for open-vocabulary inference.")
        return names

    def _validate_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(f"pixel_values must be a torch.Tensor, got {type(pixel_values)!r}")
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width]. "
                f"Received shape: {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[1] != self.config.num_channels:
            raise ValueError(
                f"Expected {self.config.num_channels} channels, received {pixel_values.shape[1]}."
            )
        if not pixel_values.is_floating_point():
            pixel_values = pixel_values.float()
        return pixel_values

    def _empty_output(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            None,
        )

    def _convert_results(self, results: Any, device: torch.device) -> YOLOEOutput:
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor | None] = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)

            if boxes is None or getattr(boxes, "xyxy", None) is None:
                box_tensor, score_tensor, label_tensor, mask_tensor = self._empty_output(device)
                boxes_list.append(box_tensor)
                scores_list.append(score_tensor)
                labels_list.append(label_tensor)
                masks_list.append(mask_tensor)
                continue

            xyxy = boxes.xyxy.detach().to(device=device, dtype=torch.float32)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)

            if conf is None:
                conf = torch.empty((xyxy.shape[0],), dtype=torch.float32, device=device)
            else:
                conf = conf.detach().to(device=device, dtype=torch.float32)

            if cls is None:
                cls = torch.empty((xyxy.shape[0],), dtype=torch.long, device=device)
            else:
                cls = cls.detach().to(device=device, dtype=torch.long)

            mask_tensor = None
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_tensor = masks.data.detach().to(device=device)

            boxes_list.append(xyxy)
            scores_list.append(conf)
            labels_list.append(cls)
            masks_list.append(mask_tensor)

        return YOLOEOutput(
            boxes=boxes_list,
            scores=scores_list,
            labels=labels_list,
            masks=masks_list,
            raw_results=results,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        class_names: list[str] | tuple[str, ...] | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> YOLOEOutput | tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor | None],
    ]:
        pixel_values = self._validate_pixel_values(pixel_values)
        backend = self._get_backend()
        backend = backend.to(pixel_values.device)
        backend.eval()

        names = self._prepare_class_names(class_names)
        if names:
            backend.set_classes(names)

        results = backend.predict(
            source=pixel_values,
            conf=self.config.score_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
            **kwargs,
        )
        output = self._convert_results(results, pixel_values.device)

        if not return_dict:
            return output.boxes, output.scores, output.labels, output.masks
        return output
