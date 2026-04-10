from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig, AutoModel, AutoProcessor

from .yolo26e import YOLOEConfig, YOLOEModel


@dataclass(frozen=True, slots=True)
class TransformersRegistration:
    model_type: str
    config_class: type
    model_class: type
    processor_class: type | None = None


_REGISTERED = False
_REGISTRATIONS = (
    TransformersRegistration(
        model_type=YOLOEConfig.model_type,
        config_class=YOLOEConfig,
        model_class=YOLOEModel,
    ),
)


def ensure_model_registrations() -> None:
    global _REGISTERED

    if _REGISTERED:
        return

    for registration in _REGISTRATIONS:
        AutoConfig.register(
            registration.model_type,
            registration.config_class,
            exist_ok=True,
        )
        AutoModel.register(
            registration.config_class,
            registration.model_class,
            exist_ok=True,
        )
        if registration.processor_class is not None:
            AutoProcessor.register(
                registration.config_class,
                registration.processor_class,
                exist_ok=True,
            )

    _REGISTERED = True
