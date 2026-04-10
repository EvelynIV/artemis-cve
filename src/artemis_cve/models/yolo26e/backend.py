from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers.utils.hub import cached_file

from .configuration import YOLOEConfig


class YOLOEBackend:
    asset_prefix = "__asset__."

    def __init__(self, config: YOLOEConfig) -> None:
        self.config = config
        self.backend = None
        self.backend_error: Exception | None = None
        self.weights_path: Path | None = None
        self._initialized = False

    def get(self) -> Any:
        if not self._initialized:
            self._initialize()

        if self.backend is None:
            reason = (
                f"{type(self.backend_error).__name__}: {self.backend_error}"
                if self.backend_error is not None
                else "unknown error"
            )
            raise RuntimeError(
                "Failed to initialize the Ultralytics YOLOE backend. "
                f"Reason: {reason}"
            )
        return self.backend

    def _initialize(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.weights_path = self._resolve_weights_path()

        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            self.backend_error = exc
            return

        if self.weights_path is None:
            self.backend_error = FileNotFoundError(
                "Could not find YOLOE weights. Expected model.safetensors "
                "next to the model config."
            )
            return

        try:
            task = self._to_ultralytics_task(self.config.task)
            self.backend = YOLOE(f"{self.config.variant}.yaml", task=task)
            self._load_safetensors_weights(self.backend, self.weights_path)
        except Exception as exc:
            self.backend_error = exc
            self.backend = None

    def _resolve_weights_path(self) -> Path | None:
        candidate = Path(self.config.name_or_path) / "model.safetensors"
        if candidate.exists():
            return candidate

        try:
            resolved = cached_file(
                self.config.name_or_path,
                "model.safetensors",
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except Exception:
            resolved = None

        return Path(resolved) if resolved else None

    @staticmethod
    def _to_ultralytics_task(task: str | None) -> str | None:
        if task is None:
            return None
        mapping = {
            "object-detection": "detect",
            "detection": "detect",
            "instance-segmentation": "segment",
            "segmentation": "segment",
        }
        return mapping.get(task, task)

    def _load_safetensors_weights(self, backend: Any, weights_path: Path) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading model.safetensors requires the 'safetensors' package."
            ) from exc

        raw_tensors = load_file(str(weights_path))
        state_dict = {
            key: value
            for key, value in raw_tensors.items()
            if not key.startswith(self.asset_prefix)
        }
        asset_tensors = {
            key: value
            for key, value in raw_tensors.items()
            if key.startswith(self.asset_prefix)
        }
        if asset_tensors:
            self._restore_embedded_assets(asset_tensors)

        model = getattr(backend, "model", None)
        if model is None:
            raise RuntimeError(
                "Ultralytics backend did not expose a 'model' module for weight loading."
            )

        incompatible = model.load_state_dict(state_dict, strict=False)
        loaded_keys = len(state_dict) - len(incompatible.unexpected_keys)
        if loaded_keys == 0:
            raise RuntimeError(
                "No tensors from model.safetensors matched the Ultralytics YOLOE model. "
                "Check that config.variant matches the checkpoint architecture."
            )

    def _restore_embedded_assets(self, asset_tensors: dict[str, torch.Tensor]) -> None:
        from ultralytics.utils import SETTINGS

        weights_dir = Path(SETTINGS["weights_dir"])
        weights_dir.mkdir(parents=True, exist_ok=True)

        for key, tensor in asset_tensors.items():
            target = weights_dir / key.removeprefix(self.asset_prefix)
            data = tensor.detach().cpu().contiguous().numpy().tobytes()
            if target.exists() and target.stat().st_size == len(data):
                continue
            target.write_bytes(data)
