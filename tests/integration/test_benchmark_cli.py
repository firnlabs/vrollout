"""Smoke test for the engine benchmark CLI."""

import re

from typer.testing import CliRunner

from benchmarks.run_engine_benchmark import app


def test_benchmark_cli_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    clean_stdout = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.stdout)
    assert result.exit_code == 0
    assert "--preset" in clean_stdout
    assert "--profile-dim" in clean_stdout
    assert "--profile-layers" in clean_stdout
