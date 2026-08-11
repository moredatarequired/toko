"""Regenerate README example outputs by running the documented commands."""

import os
import pty
import re
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
# Pinned rather than taken from $SHELL: a different shell can print different prompts
# and startup noise, which would churn every regenerated block.
SHELL = "/bin/zsh"
ALLOWED_LANGUAGES = {"txt", "json", "csv", "tsv"}
ANSI_RE = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")
# toko reports a skipped file, an unusable provider or a bad config on stderr and still
# exits 0, and pty.spawn folds stderr into the captured output. Pasting that into the
# README would document a failure as though it were the command's real result.
FAILURE_PREFIXES = ("Warning:", "Error", "Traceback (most recent call last):")


def _prepare_command(command: str) -> str:
    return re.sub(r"(?<![\w/])toko\b", "uv run toko", command)


def run_command(command: str) -> str:
    env_overrides: dict[str, str] = {"RICH_NO_COLOR": "1"}
    config_root = REPO_ROOT / ".tmp-readme-config"
    config_root.mkdir(parents=True, exist_ok=True)
    env_overrides["XDG_CONFIG_HOME"] = str(config_root)

    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()
    )
    shell_command = f"{env_prefix} {command}" if env_prefix else command

    chunks: list[str] = []

    def reader(fd: int) -> bytes:
        # pty._copy treats an empty return as EOF and stops copying, so returning b"" to
        # suppress the echo truncated the capture to whatever arrived in the first read.
        data = os.read(fd, 1024)
        chunks.append(data.decode("utf-8", "ignore"))
        return data

    pty.spawn([SHELL, "-lc", shell_command], reader)
    output = "".join(chunks)
    output = ANSI_RE.sub("", output)
    return output.rstrip("\r\n")


def failure_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith(FAILURE_PREFIXES)]


def iter_code_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("```"):
            lang = lines[i][3:].strip()
            start_index = i + 1
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
            end_index = i
            blocks.append((start_index - 1, end_index, lang))
        i += 1
    return blocks


def update_readme() -> None:
    # pty.spawn reports a missing shell as a traceback written to the pty, which would be
    # captured and pasted into the tracked README as if it were command output.
    if not os.access(SHELL, os.X_OK):
        raise RuntimeError(f"{SHELL} is required to regenerate the README examples")

    lines = README_PATH.read_text().splitlines()
    code_blocks = list(iter_code_blocks(lines))

    command_blocks = [
        (start, end, lang) for start, end, lang in code_blocks if lang == "sh"
    ]

    line_index_to_block = {
        start: (start, end, lang) for start, end, lang in code_blocks
    }

    failures: list[str] = []

    for start, end, _lang in command_blocks:
        command = "\n".join(lines[start + 1 : end])
        next_block_start = end + 1
        while next_block_start < len(lines) and not lines[next_block_start].startswith(
            "```"
        ):
            if lines[next_block_start].strip():
                next_block_start = -1
                break
            next_block_start += 1
        if next_block_start == -1 or next_block_start >= len(lines):
            continue
        next_block = line_index_to_block.get(next_block_start)
        if not next_block:
            continue
        nb_start, nb_end, nb_lang = next_block
        if nb_lang not in ALLOWED_LANGUAGES:
            continue

        output = run_command(_prepare_command(command))
        problems = failure_lines(output)
        if problems:
            failures.append("\n".join([f"$ {command}", *problems]))
            continue
        lines[nb_start + 1 : nb_end] = output.splitlines()

    if failures:
        raise RuntimeError(
            "Refusing to rewrite the README: these examples reported a failure instead "
            "of the output they document.\n\n" + "\n\n".join(failures)
        )

    README_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    update_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
