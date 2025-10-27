"""Tests for the CLI."""

from typer.testing import CliRunner

from toko.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "toko version" in result.stdout


def test_list_models():
    result = runner.invoke(app, ["--list-models"])
    assert result.exit_code == 0
    assert "openai" in result.stdout.lower()
    assert "gpt-5" in result.stdout


def test_count_with_text():
    result = runner.invoke(app, ["--text", "hello world"])
    assert result.exit_code == 0
    assert "2" in result.stdout or "token" in result.stdout.lower()


def test_count_from_stdin():
    result = runner.invoke(app, [], input="hello world")
    assert result.exit_code == 0
    assert "2" in result.stdout or "token" in result.stdout.lower()


def test_count_with_multiple_models():
    result = runner.invoke(
        app,
        ["--model", "gpt-5", "--model", "gpt-5-mini", "--text", "hello"],
    )
    assert result.exit_code == 0
    assert "gpt-5" in result.stdout
    assert "gpt-5-mini" in result.stdout
