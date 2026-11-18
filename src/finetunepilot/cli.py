"""Typer command-line interface: `finetunepilot <command>`."""

import typer

from finetunepilot import __version__

app = typer.Typer(
    name="finetunepilot",
    help="Plan, run, diagnose and compare QLoRA fine-tuning experiments.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """FineTunePilot command-line interface."""
    # An explicit callback keeps `version` as a subcommand even while it is the only one.


@app.command()
def version() -> None:
    """Print the installed FineTunePilot version."""
    typer.echo(f"finetunepilot {__version__}")


if __name__ == "__main__":
    app()
