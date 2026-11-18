"""Run comparison (pure function + rendered table + CLI)."""

from rich.console import Console
from typer.testing import CliRunner

from finetunepilot.cli import app
from finetunepilot.runlog import RunLog, write_run
from finetunepilot.tools.compare import compare_metrics, render_comparison

runner = CliRunner()


def _eval_run(run_id: str, dry_run: bool = False, **metrics) -> RunLog:
    return RunLog(
        run_id=run_id, kind="eval", name=run_id, status="success", dry_run=dry_run, metrics=metrics
    )


def test_compare_metrics_orders_and_computes_delta() -> None:
    rows = compare_metrics(
        [
            {"args_exact_match": 0.2, "json_valid": 0.5, "n": 20, "zeta": 1},
            {"args_exact_match": 0.9, "json_valid": 1.0, "n": 20, "zeta": 3},
        ]
    )
    assert [r["metric"] for r in rows] == ["json_valid", "args_exact_match", "n", "zeta"]
    by = {r["metric"]: r for r in rows}
    assert by["args_exact_match"]["delta"] == 0.7
    assert by["n"]["delta"] == 0
    assert by["zeta"]["values"] == [1, 3]


def test_compare_metrics_handles_missing_and_non_scalar() -> None:
    rows = compare_metrics(
        [
            {"final_loss": 1.5, "per_tool": {"x": {}}, "flag": True},
            {"tool_name_accuracy": 0.8, "peak_gpu_memory_mb": None},
        ]
    )
    by = {r["metric"]: r for r in rows}
    assert "per_tool" not in by and "flag" not in by  # nested and bool values are skipped
    assert by["final_loss"] == {"metric": "final_loss", "values": [1.5, None], "delta": None}
    assert by["tool_name_accuracy"]["values"] == [None, 0.8]
    assert by["peak_gpu_memory_mb"]["delta"] is None


def _render(table) -> str:
    console = Console(width=200, record=True)
    console.print(table)
    return console.export_text()


def test_render_comparison_table() -> None:
    text = _render(
        render_comparison(
            [_eval_run("base", args_exact_match=0.25), _eval_run("tuned", args_exact_match=0.9)]
        )
    )
    assert "1. base" in text and "2. tuned" in text
    assert "args_exact_match" in text and "0.2500" in text and "0.9000" in text and "0.6500" in text
    assert "dry run" in text and "random tiny model" not in text


def test_render_comparison_warns_about_dry_runs() -> None:
    table = render_comparison([_eval_run("a", dry_run=True, n=1), _eval_run("b", n=1)])
    assert "random tiny model" in str(table.caption)


def test_cli_compare(tmp_path) -> None:
    a = write_run(_eval_run("base", json_valid=0.5, args_exact_match=0.1), tmp_path)
    b = write_run(_eval_run("tuned", json_valid=1.0, args_exact_match=0.8), tmp_path)
    result = runner.invoke(app, ["compare", str(a), str(b)], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "Run comparison" in result.output
    assert "0.7000" in result.output  # delta of args_exact_match


def test_cli_compare_errors(tmp_path) -> None:
    a = write_run(_eval_run("base"), tmp_path)
    assert runner.invoke(app, ["compare", str(a)]).exit_code == 2
    assert runner.invoke(app, ["compare", str(a), str(tmp_path / "nope.json")]).exit_code == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a run"}')
    assert runner.invoke(app, ["compare", str(a), str(bad)]).exit_code == 2
