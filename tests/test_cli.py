"""Tests for the CLI."""

from typer.testing import CliRunner

from toko.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "toko version" in result.stdout


def test_list_models():
    result = runner.invoke(app, ["count", "--list-models"])
    assert result.exit_code == 0
    assert "OpenAI:" in result.stdout
    assert "gpt-4o" in result.stdout


def test_count_with_text():
    result = runner.invoke(app, ["count", "--text", "hello world"])
    assert result.exit_code == 0
    assert "2" in result.stdout or "token" in result.stdout.lower()


def test_count_from_stdin():
    result = runner.invoke(app, ["count"], input="hello world")
    assert result.exit_code == 0
    assert "2" in result.stdout or "token" in result.stdout.lower()


def test_count_with_multiple_models():
    result = runner.invoke(
        app,
        ["count", "--model", "gpt-4o", "--model", "gpt-4o-mini", "--text", "hello"],
    )
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout
    assert "gpt-4o-mini" in result.stdout
