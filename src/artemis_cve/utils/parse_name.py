from __future__ import annotations

from pathlib import Path

import typer
from transformers import AutoConfig


def parse_class_names(class_names_file: str | None, model_dir: Path) -> list[str]:
    if class_names_file:
        class_file = Path(class_names_file).expanduser()
        if not class_file.is_file():
            raise typer.BadParameter(
                f"class names file does not exist: {class_file}"
            )

        parsed = [
            line.strip()
            for line in class_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if parsed:
            return parsed

        raise typer.BadParameter(
            f"class names file is empty: {class_file}"
        )

    config = AutoConfig.from_pretrained(str(model_dir))
    default_classes = [
        str(item).strip()
        for item in getattr(config, "default_classes", [])
        if str(item).strip()
    ]
    if default_classes:
        return default_classes

    raise typer.BadParameter(
        "class_names is required for this model. "
        "Provide --class-names-file or set CLASS_NAMES_FILE."
    )
