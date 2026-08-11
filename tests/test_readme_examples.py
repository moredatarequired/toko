"""Tests for the README example regenerator's refusal to bake failures into the README."""

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts import update_readme_examples

if TYPE_CHECKING:
    from pathlib import Path

REPO_README = update_readme_examples.README_PATH
# The pinned /bin/zsh is not present everywhere; these tests only need a login shell.
TEST_SHELL = "/bin/bash"

needs_shell = pytest.mark.skipif(
    not os.access(TEST_SHELL, os.X_OK), reason=f"{TEST_SHELL} is not available"
)


def _readme_command_containing(needle: str) -> str:
    lines = REPO_README.read_text().splitlines()
    for start, end, lang in update_readme_examples.iter_code_blocks(lines):
        command = "\n".join(lines[start + 1 : end])
        if lang == "sh" and needle in command:
            return command
    raise AssertionError(f"no sh example in {REPO_README} contains {needle!r}")


def _write_example(tmp_path, command: str) -> tuple[Path, str]:
    readme = tmp_path / "README.md"
    original = f"```sh\n{command}\n```\n\n```txt\nstale output\n```\n"
    readme.write_text(original)
    return readme, original


def _point_script_at(monkeypatch, tmp_path, readme: Path, shell: str) -> None:
    monkeypatch.setattr(update_readme_examples, "README_PATH", readme)
    # REPO_ROOT only decides where run_command puts its scratch XDG_CONFIG_HOME; moving it
    # keeps the tests from leaving an untracked directory in the repo.
    monkeypatch.setattr(update_readme_examples, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(update_readme_examples, "SHELL", shell)


def _output_block(readme: Path) -> str:
    # The txt block alone: asserting over the whole file also matches the command block.
    lines = readme.read_text().splitlines()
    start, end, _lang = next(
        block
        for block in update_readme_examples.iter_code_blocks(lines)
        if block[2] == "txt"
    )
    return "\n".join(lines[start + 1 : end])


def test_missing_shell_is_refused(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "echo hello")
    _point_script_at(monkeypatch, tmp_path, readme, str(tmp_path / "no-such-shell"))

    with pytest.raises(
        RuntimeError, match="required to regenerate the README examples"
    ):
        update_readme_examples.update_readme()

    assert readme.read_text() == original


@needs_shell
def test_command_that_warns_and_exits_zero_is_refused(tmp_path, monkeypatch):
    warning = "Warning: Failed to count tokens for claude-opus-4-5: no credentials"
    readme, original = _write_example(
        tmp_path, f"echo {warning!r} >&2; echo 'model\ttokens'"
    )
    _point_script_at(monkeypatch, tmp_path, readme, TEST_SHELL)

    with pytest.raises(RuntimeError, match="Refusing to rewrite the README"):
        update_readme_examples.update_readme()

    assert readme.read_text() == original


@needs_shell
def test_readme_cost_example_is_refused_without_an_api_key(tmp_path, monkeypatch):
    command = _readme_command_containing("--cost")
    assert "claude-opus-4-5" in command

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # A login shell can put the key back from a profile, which would let the example run.
    inherits_key = subprocess.run(  # noqa: S603
        [TEST_SHELL, "-lc", 'test -n "$ANTHROPIC_API_KEY"'],
        check=False,
        capture_output=True,
    )
    if inherits_key.returncode == 0:
        pytest.skip("the login shell supplies ANTHROPIC_API_KEY, so this example runs")

    readme, original = _write_example(tmp_path, command)
    _point_script_at(monkeypatch, tmp_path, readme, TEST_SHELL)

    # Not matched on the model name: the message echoes the command, which names it too.
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY environment variable"):
        update_readme_examples.update_readme()

    assert readme.read_text() == original


@needs_shell
def test_clean_command_output_is_written(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo 'model\ttokens'")
    _point_script_at(monkeypatch, tmp_path, readme, TEST_SHELL)

    update_readme_examples.update_readme()

    assert "model\ttokens" in _output_block(readme)
    assert "stale output" not in readme.read_text()


@needs_shell
def test_output_arriving_in_several_reads_is_captured_whole(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo first; sleep 0.2; echo second")
    _point_script_at(monkeypatch, tmp_path, readme, TEST_SHELL)

    update_readme_examples.update_readme()

    captured = _output_block(readme)
    assert "first" in captured
    assert "second" in captured
