"""Command-line entry point: `sopvision <command>`."""

from pathlib import Path

import typer

app = typer.Typer(help="Live SOP monitoring with object detection and Jev.", no_args_is_help=True)


def _not_implemented() -> None:
    typer.echo("Not implemented yet")
    raise typer.Exit(2)


@app.command()
def serve() -> None:
    """Run the dashboard server."""
    _not_implemented()


@app.command()
def calibrate(
    sop: Path = typer.Option(Path("sops/server_fan.yaml"), help="SOP config"),
    run: int = typer.Option(9, help="Recording to check on (Install_<run>)"),
    out: Path = typer.Option(Path("data/calibrate_check.jpg"), help="Where to save the check image"),
) -> None:
    """Draw the SOP's slot centres on a recording, using the trained detector to find the chassis.

    Slot centres themselves come from reference boxes: python -m training.reference_labels slots
    """
    import cv2

    from sopvision.calibrate import calibration_targets, draw_layout_check
    from sopvision.config import load_sop
    from sopvision.dataset import load_index
    from sopvision.perception.detector import YoloDetector
    from sopvision.settings import Settings
    from sopvision.sources.file import FileSource

    cfg, settings = load_sop(sop), Settings()
    if not cfg.detector.weights.exists():
        typer.echo(f"Weights not found: {cfg.detector.weights}. Train the detector first (training/train.py).", err=True)
        raise typer.Exit(1)
    index = {r.run: r for r in load_index(settings.sop_data_root)}[run]
    target = next(c for c in calibration_targets(cfg, index) if c.slot == "psu_bay_1")
    frame = next(f for f in FileSource(index.raw_path, mode="lockstep").frames() if f.t >= target.t)
    dets = YoloDetector(cfg.detector.weights, cfg.detector.classes).detect(frame)
    chassis = max((d for d in dets if d.cls == "chassis"), key=lambda d: d.conf, default=None)
    if chassis is None:
        typer.echo("No chassis detected in that frame.", err=True)
        raise typer.Exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_layout_check(frame.image, chassis.xyxy, cfg))
    typer.echo(f"check image: {out}")


@app.command("eval")
def eval_cmd() -> None:
    """Replay test and scenario videos and check the acceptance criteria."""
    _not_implemented()


@app.command()
def compare() -> None:
    """Compare a native and a Docker eval report."""
    _not_implemented()


@app.command()
def doctor() -> None:
    """Report GPU and environment status."""
    from sopvision.doctor import GpuUnavailableError, gpu_report, in_container, require_gpu

    report = gpu_report()
    container = in_container()
    typer.echo(f"cuda: {report.cuda_available}")
    typer.echo(f"gpu: {report.device_name}")
    typer.echo(f"torch: {report.torch_version}")
    typer.echo(f"in_container: {container}")
    try:
        require_gpu(container, report)
    except GpuUnavailableError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


@app.command("make-scenarios")
def make_scenarios() -> None:
    """Render the simulated-mistake scenario videos."""
    _not_implemented()
