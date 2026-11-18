"""Smoke test for the CLI entry point."""

from typer.testing import CliRunner

from finetunepilot import __version__
from finetunepilot.cli import app

runner = CliRunner()


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"finetunepilot {__version__}"
