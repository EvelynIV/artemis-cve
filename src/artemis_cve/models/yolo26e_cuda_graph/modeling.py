from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ultralytics.utils import nms

from artemis_cve.models.yolo26e import YOLOEConfig, YOLOEModel, YOLOEOutput


@dataclass(slots=True)
class _CUDAGraphState:
    graph: torch.cuda.CUDAGraph
    static_input: torch.Tensor
    static_prediction: torch.Tensor


class CUDAGraphYOLOEModel(YOLOEModel):
    def __init__(self, config: YOLOEConfig) -> None:
        super().__init__(config)
        self._active_class_names: tuple[str, ...] | None = None
        self._cuda_graph_cache: dict[tuple[Any, ...], _CUDAGraphState] = {}

    @classmethod
    def from_base(cls, model: YOLOEModel) -> CUDAGraphYOLOEModel:
        upgraded = cls(model.config)
        upgraded.runtime_backend = model.runtime_backend
        return upgraded

    def _extract_prediction_tensor(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, dict):
            for value in outputs.values():
                try:
                    return self._extract_prediction_tensor(value)
                except TypeError:
                    continue
        if isinstance(outputs, (list, tuple)):
            for value in outputs:
                try:
                    return self._extract_prediction_tensor(value)
                except TypeError:
                    continue
        raise TypeError(f"Unsupported Ultralytics forward output type: {type(outputs)!r}")

    def _configure_runtime_state(
        self,
        pixel_values: torch.Tensor,
        class_names: tuple[str, ...],
        max_det: int,
    ) -> Any:
        backend = self._get_backend()
        backend = backend.to(pixel_values.device)
        backend.eval()
        backend_model = backend.model

        if self._active_class_names != class_names:
            backend.set_classes(list(class_names))
            self._active_class_names = class_names
            self._cuda_graph_cache.clear()

        backend_model.to(device=pixel_values.device, dtype=pixel_values.dtype)
        backend_model.eval()
        backend_model.set_head_attr(max_det=max_det, agnostic_nms=True)
        return backend_model

    def _convert_predictions(
        self,
        predictions: list[torch.Tensor],
        device: torch.device,
    ) -> YOLOEOutput:
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor | None] = []

        for prediction in predictions:
            if prediction.numel() == 0:
                box_tensor, score_tensor, label_tensor, mask_tensor = self._empty_output(device)
                boxes_list.append(box_tensor)
                scores_list.append(score_tensor)
                labels_list.append(label_tensor)
                masks_list.append(mask_tensor)
                continue

            xyxy = prediction[:, :4].detach().to(device=device, dtype=torch.float32)
            conf = prediction[:, 4].detach().to(device=device, dtype=torch.float32)
            cls = prediction[:, 5].detach().to(device=device, dtype=torch.long)
            mask_tensor = None
            if prediction.shape[1] > 6:
                mask_tensor = prediction[:, 6:].detach().to(device=device, dtype=torch.float32)

            boxes_list.append(xyxy)
            scores_list.append(conf)
            labels_list.append(cls)
            masks_list.append(mask_tensor)

        return YOLOEOutput(
            boxes=boxes_list,
            scores=scores_list,
            labels=labels_list,
            masks=masks_list,
            raw_results=predictions,
        )

    def _capture_cuda_graph(
        self,
        pixel_values: torch.Tensor,
        class_names: tuple[str, ...],
        max_det: int,
    ) -> _CUDAGraphState:
        backend_model = self._configure_runtime_state(
            pixel_values=pixel_values,
            class_names=class_names,
            max_det=max_det,
        )
        static_input = torch.empty_like(pixel_values, memory_format=torch.contiguous_format)
        static_input.copy_(pixel_values)

        warmup_stream = torch.cuda.Stream(device=pixel_values.device)
        current_stream = torch.cuda.current_stream(device=pixel_values.device)
        warmup_stream.wait_stream(current_stream)

        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._extract_prediction_tensor(backend_model(static_input))

        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_prediction = self._extract_prediction_tensor(backend_model(static_input))

        return _CUDAGraphState(
            graph=graph,
            static_input=static_input,
            static_prediction=static_prediction,
        )

    def _run_prediction(
        self,
        pixel_values: torch.Tensor,
        class_names: tuple[str, ...],
        max_det: int,
    ) -> torch.Tensor:
        backend_model = self._configure_runtime_state(
            pixel_values=pixel_values,
            class_names=class_names,
            max_det=max_det,
        )
        if not pixel_values.is_cuda:
            return self._extract_prediction_tensor(backend_model(pixel_values))

        graph_key = (
            pixel_values.device.index,
            pixel_values.dtype,
            tuple(pixel_values.shape),
            class_names,
            max_det,
        )
        graph_state = self._cuda_graph_cache.get(graph_key)
        if graph_state is None:
            graph_state = self._capture_cuda_graph(
                pixel_values=pixel_values,
                class_names=class_names,
                max_det=max_det,
            )
            self._cuda_graph_cache[graph_key] = graph_state

        graph_state.static_input.copy_(pixel_values)
        graph_state.graph.replay()
        return graph_state.static_prediction

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
        pixel_values = self._validate_pixel_values(pixel_values).contiguous()
        names = tuple(self._prepare_class_names(class_names))
        requested_max_det = kwargs.pop("max_det", None)
        max_det = int(requested_max_det if requested_max_det is not None else 300)
        if max_det <= 0:
            max_det = 300

        prediction = self._run_prediction(
            pixel_values=pixel_values,
            class_names=names,
            max_det=max_det,
        )
        backend_model = self._get_backend().model
        filtered_predictions = nms.non_max_suppression(
            prediction=prediction,
            conf_thres=float(self.config.score_threshold),
            iou_thres=float(self.config.iou_threshold),
            agnostic=True,
            max_det=max_det,
            nc=len(names),
            end2end=bool(getattr(backend_model, "end2end", False)),
        )
        output = self._convert_predictions(filtered_predictions, pixel_values.device)

        if not return_dict:
            return output.boxes, output.scores, output.labels, output.masks
        return output
