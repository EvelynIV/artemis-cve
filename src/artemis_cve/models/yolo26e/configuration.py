from __future__ import annotations

from transformers import PretrainedConfig


class YOLOEConfig(PretrainedConfig):
    model_type = "yoloe"

    def __init__(
        self,
        variant: str = "yoloe-26x-seg",
        task: str = "instance-segmentation",
        image_size: int = 640,
        num_channels: int = 3,
        segmentation: bool = True,
        open_vocab: bool = True,
        default_classes: list[str] | None = None,
        num_labels: int | None = None,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        stride: list[int] | None = None,
        model_input_name: str = "pixel_values",
        dtype: str = "float32",
        torch_dtype: str | None = None,
        **kwargs,
    ) -> None:
        default_classes = default_classes or []
        stride = stride or [8, 16, 32]
        resolved_dtype = torch_dtype or dtype

        if id2label is None:
            id2label = {idx: name for idx, name in enumerate(default_classes)}
        else:
            id2label = {int(idx): name for idx, name in id2label.items()}

        if label2id is None:
            label2id = {name: idx for idx, name in id2label.items()}

        if num_labels is None:
            num_labels = len(id2label)

        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            dtype=resolved_dtype,
            **kwargs,
        )

        self.variant = variant
        self.task = task
        self.image_size = image_size
        self.num_channels = num_channels
        self.segmentation = segmentation
        self.open_vocab = open_vocab
        self.default_classes = list(default_classes)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.stride = list(stride)
        self.model_input_name = model_input_name
        self.dtype = resolved_dtype
