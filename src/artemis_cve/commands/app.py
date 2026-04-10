from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import grpc
import typer
from transformers import AutoConfig

from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artemis_cve.servicers import WebRtcDetectorServicer

app = typer.Typer(name="artemis-cve")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "model-bin" / "hf_yoloe"


def _parse_class_names(raw: str | None, model_dir: Path) -> list[str]:
    if raw:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if parsed:
            return parsed

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    default_classes = [str(item).strip() for item in getattr(config, "default_classes", []) if str(item).strip()]
    if default_classes:
        return default_classes

    raise typer.BadParameter(
        "class_names is required for this model. "
        "Provide --class-names or set ARTEMIS_CVE_CLASS_NAMES."
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", envvar="ARTEMIS_CVE_HOST", help="Listen address."),
    port: int = typer.Option(50051, envvar="ARTEMIS_CVE_PORT", help="Listen port."),
    model_dir: str = typer.Option(
        str(DEFAULT_MODEL_DIR),
        envvar="ARTEMIS_CVE_MODEL_DIR",
        help="Local YOLOE model directory.",
    ),
    device: str = typer.Option(
        "cpu",
        envvar="ARTEMIS_CVE_DEVICE",
        help="Inference device, for example cpu or cuda:0.",
    ),
    class_names: str = typer.Option(
        "",
        envvar="ARTEMIS_CVE_CLASS_NAMES",
        help="Comma-separated open-vocabulary class names.",
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    resolved_model_dir = Path(model_dir).resolve()
    resolved_class_names = _parse_class_names(class_names, resolved_model_dir)

    async def _run() -> None:
        server = grpc.aio.server()
        pb2_grpc.add_WebRtcDetectorEngineServicer_to_server(
            WebRtcDetectorServicer(
                model_dir=str(resolved_model_dir),
                class_names=resolved_class_names,
                device=device,
            ),
            server,
        )
        listen_addr = f"{host}:{port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        typer.echo(f"gRPC server listening on {listen_addr}")
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
