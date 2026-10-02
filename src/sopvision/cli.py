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
    _not_implemented()


@app.command("make-scenarios")
def make_scenarios() -> None:
    """Render the simulated-mistake scenario videos."""
    _not_implemented()
