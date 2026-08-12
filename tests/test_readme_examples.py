"""Tests for the README example regenerator's refusal to bake failures into the README."""

import os
import subprocess
from pathlib import Path

import pytest

from scripts import update_readme_examples

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


def _write_readme(tmp_path, examples: list[tuple[str, str]]) -> tuple[Path, str]:
    readme = tmp_path / "README.md"
    original = "\n".join(
        f"```sh\n{command}\n```\n\n```txt\n{output}\n```\n"
        for command, output in examples
    )
    readme.write_text(original)
    return readme, original


def _write_example(tmp_path, command: str) -> tuple[Path, str]:
    return _write_readme(tmp_path, [(command, "stale output")])


def _point_script_at(monkeypatch, readme: Path, shell: str) -> None:
    monkeypatch.setattr(update_readme_examples, "README_PATH", readme)
    monkeypatch.setattr(update_readme_examples, "SHELL", shell)


def _blocks(readme: Path, language: str) -> list[str]:
    # The fenced blocks alone: asserting over the whole file cannot tell a regenerated
    # output block from the command block that produced it.
    lines = readme.read_text().splitlines()
    return [
        "\n".join(lines[block.start + 1 : block.end])
        for block in update_readme_examples.iter_code_blocks(lines)
        if block.language == language
    ]


def _commands(readme: Path) -> list[str]:
    return _blocks(readme, "sh")


def _output_block(readme: Path) -> str:
    return _blocks(readme, "txt")[0]


def _reasons(stale: list[update_readme_examples.StaleExample]) -> list[str]:
    # The reasons alone: a report also echoes the command, which repeats the very text a
    # reason names, so searching the rendered report proves only that some reason fired.
    return [reason for example in stale for reason in example.reasons]


def test_only_a_bare_toko_command_is_rewritten():
    prepare = update_readme_examples._prepare_command  # noqa: SLF001

    assert prepare("toko --text hi") == "uv run -q toko --text hi"
    assert prepare("printf hi | toko") == "printf hi | uv run -q toko"
    assert prepare("ls ~/.cache/toko") == "ls ~/.cache/toko"
    assert prepare("ls my-toko") == "ls my-toko"
    assert prepare("ls dir.toko") == "ls dir.toko"


@needs_shell
def test_a_warning_from_the_launcher_is_not_blamed_on_toko(tmp_path, monkeypatch):
    # A real `uv run`, given the mismatched VIRTUAL_ENV that uv warns about. The warning is
    # uv's, not toko's, so it must not leave the example stale.
    readme, _ = _write_example(tmp_path, "VIRTUAL_ENV=/no-such-venv toko --version")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.update_readme() == []

    assert "toko version" in _output_block(readme)


def test_missing_shell_is_refused(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "echo hello")
    _point_script_at(monkeypatch, readme, str(tmp_path / "no-such-shell"))

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
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert _reasons(stale) == [warning]
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
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # A reason, not the whole report: the report echoes the command, and matching the
    # echo would prove only that the example failed, not that the key is why. The reason
    # is toko's own line, so the surrounding wording is left free.
    assert any(
        "ANTHROPIC_API_KEY environment variable" in reason for reason in _reasons(stale)
    )
    # Exactly one: the missing key is the only complaint, so noise from the launcher
    # creeping back into the reasons would fail here rather than pass the check above.
    assert len(_reasons(stale)) == 1
    assert readme.read_text() == original


@needs_shell
def test_clean_command_output_is_written(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo 'model\ttokens'")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.update_readme() == []

    assert "model\ttokens" in _output_block(readme)
    assert "stale output" not in readme.read_text()


@needs_shell
def test_output_arriving_in_several_reads_is_captured_whole(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo first; sleep 0.2; echo second")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    update_readme_examples.update_readme()

    captured = _output_block(readme)
    assert "first" in captured
    assert "second" in captured


@needs_shell
def test_a_word_that_only_starts_with_error_is_not_a_failure(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo 'Errors: 0'")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.update_readme() == []

    assert _output_block(readme) == "Errors: 0"


@needs_shell
def test_an_error_message_is_still_a_failure(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "echo 'Error fetching URL: 404'")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert _reasons(stale) == ["Error fetching URL: 404"]
    assert readme.read_text() == original


@needs_shell
def test_a_command_that_prints_nothing_is_a_failure(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "true")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert _reasons(stale) == ["the command printed nothing"]
    assert readme.read_text() == original


@needs_shell
def test_a_command_that_prints_only_whitespace_is_a_failure(tmp_path, monkeypatch):
    """Otherwise the block is silently emptied, which is what the guard exists to stop."""
    readme, original = _write_example(tmp_path, "printf '   '")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert _reasons(stale) == ["the command printed nothing"]
    assert readme.read_text() == original


@needs_shell
def test_an_indented_warning_is_still_a_failure(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "echo '   Warning: indented'")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # The indentation survives: the match is made on the stripped line, but the reason
    # quotes the line as printed.
    assert _reasons(stale) == ["   Warning: indented"]
    assert readme.read_text() == original


@needs_shell
def test_a_traceback_is_a_failure(tmp_path, monkeypatch):
    readme, original = _write_example(
        tmp_path, "echo 'Traceback (most recent call last):'; echo '  File \"x.py\"'"
    )
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # Only the header line is a reason; the frame below it is not matched.
    assert _reasons(stale) == ["Traceback (most recent call last):"]
    assert readme.read_text() == original


@needs_shell
def test_a_nonzero_exit_is_a_failure_however_clean_the_output(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "echo 'model\ttokens'; exit 2")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # The clean output is not a second reason; the exit code is the whole complaint.
    assert _reasons(stale) == ["the command exited 2"]
    assert readme.read_text() == original


@needs_shell
def test_a_multi_line_command_reports_only_its_reasons(tmp_path, monkeypatch):
    command = "echo one\necho two\nexit 2"
    readme, original = _write_example(tmp_path, command)
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # The command's own continuation lines are not reasons.
    assert _reasons(stale) == ["the command exited 2"]
    assert _commands(readme) == [command]
    assert readme.read_text() == original


@needs_shell
def test_a_command_the_shell_cannot_find_is_a_failure(tmp_path, monkeypatch):
    readme, original = _write_example(tmp_path, "no-such-command-xyz --version")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    # The shell's own "command not found" is not a reason: it matches no failure pattern.
    assert _reasons(stale) == ["the command exited 127"]
    assert readme.read_text() == original


@needs_shell
def test_a_growing_block_does_not_hide_a_later_failure(tmp_path, monkeypatch):
    # Every example is found and run before anything is rewritten, so an earlier block
    # whose output changes length cannot move a later one out from under the guard.
    later = "echo 'Warning: bad' >&2; echo fine"
    readme, _ = _write_readme(tmp_path, [("seq 1 3", "one line"), (later, "second")])
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert _reasons(stale) == ["Warning: bad"]
    assert _commands(readme) == ["seq 1 3", later]
    assert _blocks(readme, "txt") == ["1\n2\n3", "second"]


@needs_shell
def test_a_growing_block_does_not_overwrite_a_later_command(tmp_path, monkeypatch):
    readme, _ = _write_readme(
        tmp_path, [("seq 1 5", "one line"), ("echo second", "stale")]
    )
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.update_readme() == []

    assert _commands(readme) == ["seq 1 5", "echo second"]
    assert _blocks(readme, "txt") == ["1\n2\n3\n4\n5", "second"]


@needs_shell
def test_a_failing_example_leaves_the_others_regenerated(tmp_path, monkeypatch):
    readme, _ = _write_readme(
        tmp_path, [("echo 'Error: nope'", "keep me"), ("echo fresh", "stale")]
    )
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    stale = update_readme_examples.update_readme()

    assert len(stale) == 1
    assert stale[0].reasons == ["Error: nope"]
    assert _blocks(readme, "txt") == ["keep me", "fresh"]


@needs_shell
def test_the_examples_run_isolated_from_the_developers_own_caches(
    tmp_path, monkeypatch
):
    readme, _ = _write_example(tmp_path, "printenv XDG_CACHE_HOME")
    _point_script_at(monkeypatch, readme, TEST_SHELL)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "developers-cache"))

    assert update_readme_examples.update_readme() == []

    used_cache = _output_block(readme)
    assert "developers-cache" not in used_cache
    # The scratch home is a temporary directory, so nothing is left behind to clean up.
    assert not Path(used_cache).exists()


@needs_shell
def test_the_examples_run_isolated_from_the_developers_own_config(
    tmp_path, monkeypatch
):
    readme, _ = _write_example(tmp_path, "printenv XDG_CONFIG_HOME")
    _point_script_at(monkeypatch, readme, TEST_SHELL)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "developers-config"))

    assert update_readme_examples.update_readme() == []

    used_config = _output_block(readme)
    assert "developers-config" not in used_config
    assert not Path(used_config).exists()


@needs_shell
def test_the_isolation_reaches_every_command_of_a_pipeline(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "printf 'x' | printenv XDG_CACHE_HOME")
    _point_script_at(monkeypatch, readme, TEST_SHELL)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "developers-cache"))

    assert update_readme_examples.update_readme() == []

    used_cache = _output_block(readme)
    assert "developers-cache" not in used_cache
    assert not Path(used_cache).exists()


@needs_shell
def test_the_developers_active_virtualenv_does_not_reach_an_example(
    tmp_path, monkeypatch
):
    readme, _ = _write_example(tmp_path, 'echo "[${VIRTUAL_ENV:-unset}]"')
    _point_script_at(monkeypatch, readme, TEST_SHELL)
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "developers-venv"))

    assert update_readme_examples.update_readme() == []

    assert _output_block(readme) == "[unset]"


@needs_shell
def test_the_examples_run_from_the_repository_root(tmp_path, monkeypatch):
    """`uv run toko` resolves the project from the working directory, not from $PATH."""
    readme, _ = _write_example(tmp_path, "pwd -P")
    _point_script_at(monkeypatch, readme, TEST_SHELL)
    monkeypatch.chdir(tmp_path)

    assert update_readme_examples.update_readme() == []

    assert _output_block(readme) == str(update_readme_examples.REPO_ROOT)


@needs_shell
def test_main_exits_nonzero_when_an_example_is_left_stale(
    tmp_path, monkeypatch, capsys
):
    """The exit code is the only thing that makes the guard visible to CI or a human."""
    # printf assembles the reason, so "Error: nope" appears nowhere in the command. A
    # report that echoed only the command, or that listed only the reasons, satisfies
    # one of the assertions below but never both.
    command = "printf 'Error: %s\\n' nope"
    readme, original = _write_example(tmp_path, command)
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.main() == 1

    report = capsys.readouterr().err
    assert "Error: nope" in report
    assert f"$ {command}" in report
    assert readme.read_text() == original


@needs_shell
def test_main_exits_zero_when_every_example_regenerates(tmp_path, monkeypatch):
    readme, _ = _write_example(tmp_path, "echo fresh")
    _point_script_at(monkeypatch, readme, TEST_SHELL)

    assert update_readme_examples.main() == 0

    assert _output_block(readme) == "fresh"


def test_main_exits_nonzero_when_the_shell_is_missing(tmp_path, monkeypatch, capsys):
    readme, original = _write_example(tmp_path, "echo hello")
    _point_script_at(monkeypatch, readme, str(tmp_path / "no-such-shell"))

    assert update_readme_examples.main() == 1

    assert "required to regenerate the README examples" in capsys.readouterr().err
    assert readme.read_text() == original
