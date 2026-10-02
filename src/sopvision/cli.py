"""Command-line entry point: `sopvision <command>`."""

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
def calibrate() -> None:
    """Click slot centres for an SOP layout."""
    _not_implemented()


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
