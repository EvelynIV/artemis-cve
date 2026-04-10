from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from pathlib import Path

import grpc
import typer
from transformers import AutoConfig

from artemis_cve.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artemis_cve.servicers import WebRtcDetectorServicer

app = typer.Typer(name="artemis-cve", invoke_without_command=True, no_args_is_help=False)
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


def _serve(
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
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, stop_event.set)
        typer.echo(f"gRPC server listening on {listen_addr}")
        try:
            await stop_event.wait()
        finally:
            await server.stop(grace=1)

    asyncio.run(_run())


@app.callback()
def main(
    ctx: typer.Context,
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
    if ctx.invoked_subcommand is None:
        _serve(
            host=host,
            port=port,
            model_dir=model_dir,
            device=device,
            class_names=class_names,
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
    _serve(
        host=host,
        port=port,
        model_dir=model_dir,
        device=device,
        class_names=class_names,
    )


if __name__ == "__main__":
    app()
