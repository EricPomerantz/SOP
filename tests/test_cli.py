from typer.testing import CliRunner

from sopvision.cli import app


def test_cli_lists_all_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["serve", "calibrate", "eval", "compare", "doctor", "make-scenarios"]:
        assert cmd in result.output
