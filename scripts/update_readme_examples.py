"""Regenerate README example outputs by running the documented commands."""

import contextlib
import os
import pty
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
# Pinned rather than taken from $SHELL: a different shell can print different prompts
# and startup noise, which would churn every regenerated block.
SHELL = "/bin/zsh"
ALLOWED_LANGUAGES = {"txt", "json", "csv", "tsv"}
ANSI_RE = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
# toko reports a skipped file, an unusable provider or a bad config on stderr and still
# exits 0, and the pty folds stderr into the captured output. Pasting that into the README
# would document a failure as though it were the command's real result. "error" is matched
# on a word boundary so that a real output line such as "Errors: 0" is not a failure.
FAILURE_RE = re.compile(
    r"^(?:warning:|error\b|traceback \(most recent call last\):)", re.IGNORECASE
)


class CodeBlock(NamedTuple):
    start: int
    end: int
    language: str


class Example(NamedTuple):
    command: str
    output: CodeBlock


class CommandResult(NamedTuple):
    output: str
    exit_code: int


class StaleExample(NamedTuple):
    command: str
    reasons: list[str]


def _prepare_command(command: str) -> str:
    # `-q` silences uv's own warnings, which FAILURE_RE would otherwise blame on toko and
    # refuse to regenerate for. A single `-q` still lets uv's errors through.
    return re.sub(r"(?<![\w./-])toko\b", "uv run -q toko", command)


def _uv_cache_dir() -> str:
    if override := os.environ.get("UV_CACHE_DIR"):
        return override
    ambient = os.environ.get("XDG_CACHE_HOME")
    base = Path(ambient) if ambient else Path.home() / ".cache"
    return str(base / "uv")


def run_command(command: str, scratch: Path) -> CommandResult:
    config_home = scratch / "config"
    cache_home = scratch / "cache"
    for directory in (config_home, cache_home):
        directory.mkdir(parents=True, exist_ok=True)
    env_overrides = {
        # NO_COLOR, not RICH_NO_COLOR: rich reads the former, and the latter suppressed
        # nothing. ANSI_RE strips what does get emitted, so this only keeps the captured
        # bytes closer to what lands in the README.
        "NO_COLOR": "1",
        # Isolated so that neither the developer's own config nor counts cached from an
        # earlier run can change what an example prints.
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_CACHE_HOME": str(cache_home),
        # uv keeps its own cache under XDG_CACHE_HOME, and a cold uv cache would rebuild
        # the environment from the network on every example, so it keeps the real one.
        "UV_CACHE_DIR": _uv_cache_dir(),
    }
    env_assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
    )
    # Exported rather than written as an assignment prefix: a prefix applies only to the
    # first command of a pipeline, so `printf … | toko` would run toko unisolated. Exporting
    # inside the command string still runs after the login shell's profile, so the overrides
    # win over it. VIRTUAL_ENV goes for the same reason the XDG homes are replaced: an
    # example must not run against whatever environment the developer has active.
    script = f"unset VIRTUAL_ENV; export {env_assignments}; {command}"

    # A terminal, not a pipe: toko draws tables only when stdout is a tty, and treats a
    # non-tty stdin as piped input. openpty plus a subprocess rather than pty.spawn, whose
    # forkpty is unsafe to call from a multi-threaded parent such as a test runner.
    controller, follower = pty.openpty()
    chunks: list[bytes] = []
    try:
        try:
            process = subprocess.Popen(  # noqa: S603
                [SHELL, "-lc", script],
                # The examples say `toko`, which becomes `uv run -q toko`, and uv resolves
                # the project from the working directory. Without this the script only
                # works when invoked from the repository root.
                cwd=REPO_ROOT,
                stdin=follower,
                stdout=follower,
                stderr=follower,
            )
        finally:
            os.close(follower)
        # Reading as the command runs, so a command that outfills the pty buffer cannot
        # deadlock against us. The read raises EIO once the child closes its end.
        with contextlib.suppress(OSError):
            while data := os.read(controller, 1024):
                chunks.append(data)
        exit_code = process.wait()
    finally:
        os.close(controller)

    output = ANSI_RE.sub("", b"".join(chunks).decode("utf-8", "ignore"))
    return CommandResult(output.rstrip("\r\n"), exit_code)


def failure_reasons(result: CommandResult) -> list[str]:
    reasons: list[str] = []
    if result.exit_code != 0:
        # Catches what the layers around toko report: a shell that cannot find the
        # command, a uv that cannot build the environment, a rejected CLI flag.
        reasons.append(f"the command exited {result.exit_code}")
    if not result.output.strip():
        reasons.append("the command printed nothing")
    reasons.extend(
        line for line in result.output.splitlines() if FAILURE_RE.match(line.strip())
    )
    return reasons


def iter_code_blocks(lines: list[str]) -> list[CodeBlock]:
    blocks: list[CodeBlock] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("```"):
            language = lines[i][3:].strip()
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
            blocks.append(CodeBlock(start, i, language))
        i += 1
    return blocks


def _block_after(
    lines: list[str], end: int, by_start: dict[int, CodeBlock]
) -> CodeBlock | None:
    index = end + 1
    while index < len(lines) and not lines[index].startswith("```"):
        if lines[index].strip():
            return None
        index += 1
    return by_start.get(index)


def documented_examples(lines: list[str]) -> list[Example]:
    blocks = iter_code_blocks(lines)
    by_start = {block.start: block for block in blocks}
    examples: list[Example] = []
    for block in blocks:
        if block.language != "sh":
            continue
        output = _block_after(lines, block.end, by_start)
        if output is None or output.language not in ALLOWED_LANGUAGES:
            continue
        examples.append(Example("\n".join(lines[block.start + 1 : block.end]), output))
    return examples


def update_readme() -> list[StaleExample]:
    # Checked before anything runs, so a machine without the pinned shell gets one clear
    # message rather than a traceback out of the middle of the first example.
    if not os.access(SHELL, os.X_OK):
        raise RuntimeError(f"{SHELL} is required to regenerate the README examples")

    lines = README_PATH.read_text(encoding="utf-8").splitlines()
    examples = documented_examples(lines)

    with tempfile.TemporaryDirectory(prefix="toko-readme-") as scratch:
        results = [
            (example, run_command(_prepare_command(example.command), Path(scratch)))
            for example in examples
        ]

    regenerated: list[tuple[Example, str]] = []
    stale: list[StaleExample] = []
    for example, result in results:
        reasons = failure_reasons(result)
        if reasons:
            stale.append(StaleExample(example.command, reasons))
        else:
            regenerated.append((example, result.output))

    # Last block first: rewriting one output block moves every line below it, so working
    # backwards keeps the line numbers of the blocks still to write correct.
    for example, output in reversed(regenerated):
        lines[example.output.start + 1 : example.output.end] = output.splitlines()

    README_PATH.write_text("\n".join(lines) + "\n")
    return stale


def main() -> int:
    try:
        stale = update_readme()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    if stale:
        reports = [
            "\n".join([f"$ {example.command}", *example.reasons]) for example in stale
        ]
        print(
            f"Left {len(stale)} of the README's examples at their previous output rather "
            "than documenting a failure:\n\n" + "\n\n".join(reports),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
