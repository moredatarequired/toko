"""Run git against a fixture directory, and never against the ambient repository."""

import os
import subprocess
from collections.abc import Mapping  # noqa: TC003
from pathlib import Path  # noqa: TC003

from tests.conftest import git_vars_to_scrub


def fixture_git_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    for name in git_vars_to_scrub(env):
        env.pop(name, None)
    return env


def run_git(target: Path, *args: str) -> str:
    """Run a git command whose target directory cannot be left out.

    `target` is positional and has no default, so the omission that put a fixture's
    `git config --local` into the real repository -- a call site that simply forgot
    `cwd=` and inherited the process's own directory -- no longer type-checks or
    runs. It is passed twice, as `-C` and as `cwd`, because those two answer
    different questions: `-C` names the repository, `cwd` names the directory a
    relative path in `args` resolves against.

    The environment is scrubbed here as well as by `isolated_git_env`, so that the
    guarantee belongs to every write site that goes through this function
    rather than to an autouse fixture in another file that they never mention.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(target), *args],  # noqa: S607
        cwd=target,
        env=fixture_git_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
