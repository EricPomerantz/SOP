"""Command-line entry point: `sopvision <command>`."""

from pathlib import Path

import typer

app = typer.Typer(help="Live SOP monitoring with object detection and Jev.", no_args_is_help=True)


def _check_data_root(settings) -> None:
    from sopvision.config import ConfigError

    try:
        settings.check_data_root()
    except ConfigError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


@app.command()
def serve(host: str = typer.Option("127.0.0.1", help="Interface to listen on"),
          port: int = typer.Option(8000, help="Port")) -> None:
    """Run the dashboard server."""
    import uvicorn

    from sopvision.config import ConfigError
    from sopvision.doctor import GpuUnavailableError, gpu_report
    from sopvision.server.app import create_app
    from sopvision.server.runs import RunManager, default_run_factory, startup_checks
    from sopvision.settings import Settings

    settings = Settings()
    try:
        sops = startup_checks(settings, gpu_report())
    except (ConfigError, GpuUnavailableError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    manager = RunManager(settings, {s.sop: s for s in sops}, default_run_factory(settings))
    typer.echo(f"SOP Vision on http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}")
    uvicorn.run(create_app(settings, manager), host=host, port=port, log_level="warning")


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
    _check_data_root(settings)
    if not cfg.detector.weights.exists():
        typer.echo(f"Weights not found: {cfg.detector.weights}. Train the detector first (training/train.py).", err=True)
        raise typer.Exit(1)
    index = {r.run: r for r in load_index(settings.sop_data_root)}[run]
    targets = calibration_targets(cfg, index)
    target = targets[-2] if len(targets) > 1 else targets[-1]  # most parts in place, final step not yet done
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
def eval_cmd(out: Path = typer.Option(Path("runs/eval-native.json"), help="Where to write the eval report")) -> None:
    """Replay test and scenario videos and check the acceptance criteria."""
    from sopvision.config import ConfigError
    from sopvision.doctor import GpuUnavailableError, gpu_report, in_container
    from sopvision.evaluation.run_eval import run_eval, summary_lines
    from sopvision.server.runs import startup_checks
    from sopvision.settings import Settings

    settings, gpu = Settings(), gpu_report()
    try:
        sops = startup_checks(settings, gpu)
    except (ConfigError, GpuUnavailableError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    env = {"in_container": in_container(), "gpu": gpu.device_name, "torch": gpu.torch_version, "sop": sops[0].sop}
    report = run_eval(settings, sops[0], out, env, progress=typer.echo)
    for line in summary_lines(report):
        typer.echo(line)
    typer.echo(f"report: {out}")
    if not report.passed:
        raise typer.Exit(1)


@app.command()
def compare(first: Path = typer.Argument(..., help="Eval report, e.g. runs/eval-native.json"),
            second: Path = typer.Argument(..., help="Eval report, e.g. runs/eval-docker.json")) -> None:
    """Compare a native and a Docker eval report."""
    from sopvision.evaluation.compare import compare_reports
    from sopvision.evaluation.metrics import EvalReport

    a = EvalReport.model_validate_json(first.read_text(encoding="utf-8"))
    b = EvalReport.model_validate_json(second.read_text(encoding="utf-8"))
    problems = compare_reports(a, b)
    for p in problems:
        typer.echo(p)
    if problems:
        raise typer.Exit(1)
    typer.echo("parity: OK")


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
def make_scenarios(run: int = typer.Option(12, help="Recording to build the scenarios from"),
                   out: Path = typer.Option(Path("data/scenarios"), help="Output folder")) -> None:
    """Render the simulated-mistake scenario videos."""
    from sopvision.dataset import load_index
    from sopvision.evaluation.scenarios import SCENARIOS, render_scenario
    from sopvision.settings import Settings

    settings = Settings()
    _check_data_root(settings)
    index = {r.run: r for r in load_index(settings.sop_data_root)}[run]
    for spec in SCENARIOS:
        typer.echo(f"wrote {render_scenario(index, spec, out)}")
