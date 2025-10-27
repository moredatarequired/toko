"""Tests for the CLI."""

import os
from pathlib import Path

from typer.testing import CliRunner

from toko.cli import app

runner = CliRunner()


def _invoke_cli(args: list[str], env_overrides: dict[str, str] | None = None):
    """Invoke the CLI in an isolated filesystem with predictable config."""
    overrides = dict(env_overrides or {})
    with runner.isolated_filesystem():
        overrides.setdefault("XDG_CONFIG_HOME", str(Path.cwd()))
        return runner.invoke(app, args, env=overrides)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("toko version ")


def test_list_models():
    result = runner.invoke(app, ["--list-models"])
    assert result.exit_code == 0
    assert "openai" in result.stdout.lower()
    assert "gpt-5" in result.stdout


def test_count_with_text():
    result = _invoke_cli(["--header", "--format", "tsv", "--text", "hello world"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t2"


def test_count_from_stdin():
    result = runner.invoke(app, ["--header", "--format", "tsv"], input="hello world")
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t2"


def test_count_with_multiple_models():
    result = _invoke_cli(
        [
            "--header",
            "--format",
            "tsv",
            "--model",
            "gpt-5",
            "--model",
            "gpt-5-mini",
            "--text",
            "hello",
        ]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "model\ttokens\ngpt-5\t1\ngpt-5-mini\t1"


def test_default_no_header_when_not_tty(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: False)
    text = "header-default-pipe"
    result = _invoke_cli(["--model", "gpt-5", "--model", "gpt-5-mini", "--text", text])
    lines = [line for line in result.stdout.splitlines() if line]
    assert lines
    assert lines[0].startswith("gpt-5\t")
    assert lines[1].startswith("gpt-5-mini\t")


def test_header_flag_forces_header(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: False)
    text = "header-flag"
    result = _invoke_cli(
        ["--header", "--model", "gpt-5", "--model", "gpt-5-mini", "--text", text]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert first_line.lower().startswith("model")


def test_tty_default_includes_header(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    text = "tty-header"
    result = _invoke_cli(
        ["--format", "tsv", "--model", "gpt-5", "--model", "gpt-5-mini", "--text", text]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert first_line.lower().startswith("model")
    assert any(line.startswith("gpt-5\t") for line in result.stdout.splitlines())


def test_no_header_flag_respected(monkeypatch):
    monkeypatch.setattr("toko.cli.is_stdout_tty", lambda: True)
    text = "tty-no-header"
    result = _invoke_cli(
        [
            "--no-header",
            "--format",
            "tsv",
            "--model",
            "gpt-5",
            "--model",
            "gpt-5-mini",
            "--text",
            text,
        ]
    )
    first_line = next((line for line in result.stdout.splitlines() if line), "")
    assert not first_line.lower().startswith("model")


def test_partial_success_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")

    text = "test-missing-anthropic-partial"
    result = _invoke_cli(
        ["--model", "gpt-5", "--model", "claude-sonnet-4-5", "--text", text]
    )
    assert result.exit_code == 0
    assert "gpt-5" in result.stdout
    assert "claude-sonnet-4-5" not in result.stdout
    assert "Failed to count tokens for claude-sonnet-4-5" in result.stderr
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_all_fail_missing_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not os.environ.get("ANTHROPIC_API_KEY")

    text = "test-missing-anthropic-all"
    result = _invoke_cli(["--model", "claude-sonnet-4-5", "--text", text])
    assert result.exit_code != 0
    assert "Error: All models failed to count tokens" in result.stderr


def test_partial_success_missing_google_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not os.environ.get("GOOGLE_API_KEY")

    text = "test-missing-google-partial"
    result = _invoke_cli(
        ["--model", "gpt-5", "--model", "models/gemini-2.5-flash", "--text", text]
    )
    assert result.exit_code == 0
    assert "gpt-5" in result.stdout
    assert "gemini-2.5-flash" not in result.stdout
    assert "GOOGLE_API_KEY" in result.stderr


def test_partial_success_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not os.environ.get("HF_TOKEN")

    text = "test-missing-hf-token-partial"
    result = _invoke_cli(
        ["--model", "gpt-5", "--model", "meta-llama/Llama-3.2-1B", "--text", text]
    )
    assert result.exit_code == 0
    assert "gpt-5" in result.stdout
    assert "meta-llama/Llama-3.2-1B" not in result.stdout
    assert "Failed to count tokens for meta-llama/Llama-3.2-1B" in result.stderr
    assert "HF_TOKEN" in result.stderr
