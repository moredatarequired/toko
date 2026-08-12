"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work.

Run it as `python tests/smoke_test.py`, which is how the release workflow invokes it
against an installed artifact. Importing it runs no check, creates no directory and
redirects no cache — which matters because the file matches pytest's default
`*_test.py` glob, so an ordinary collection imports it.
"""

import contextlib
import io
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import toko
from toko.cli import app
from toko.counter import count_tokens
from toko.models import list_models

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def isolated_cache_home() -> Iterator[None]:
    """Point the cache and config dirs at a throwaway tree for the duration.

    The checks below dispatch clear-cache, which deletes the token cache and the cached
    prices it finds, so this has to wrap them: nothing under toko resolves a cache path
    at import time, but the first call that touches one must already see this. Both
    variables are consulted ahead of the platform default on every platform, and
    `setdefault` leaves a caller — CI sharing one download cache, say — free to say
    where instead.
    """
    tmp = tempfile.mkdtemp(prefix="toko-smoke-")
    names = ("XDG_CACHE_HOME", "XDG_CONFIG_HOME")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.setdefault(name, tmp)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    with isolated_cache_home():
        assert toko.__version__ is not None
        assert toko.__version__ != "0.0.0"
        assert count_tokens("The most basic of smoke tests.", model="gpt-5").count == 7

        models_by_provider = list_models()
        assert len(models_by_provider) > 4
        assert len(models_by_provider["anthropic"]) > 5
        assert len(models_by_provider["openai"]) > 5

        # Running the CLI, not just importing it, is what catches a dependency that toko
        # imports but never declares — the failure mode typer 0.26 exposed by dropping
        # click.
        version_output = io.StringIO()
        with (
            contextlib.redirect_stdout(version_output),
            contextlib.suppress(SystemExit),
        ):
            app(["--version"])
        assert toko.__version__ in version_output.getvalue()

        # --version is an eager no-value flag, so it never reaches the option scan
        # TokoGroup uses to find a subcommand past a global option. Dispatch a subcommand
        # behind a separated option value instead: if a future typer changes how params
        # are built, the scan stops recognising -m as value-taking and "clear-cache" is
        # read as a path.
        dispatch_output = io.StringIO()
        with (
            contextlib.redirect_stdout(dispatch_output),
            contextlib.suppress(SystemExit),
        ):
            app(["-m", "gpt-5", "clear-cache"])
        assert "Cache cleared" in dispatch_output.getvalue()

    print("Smoke test succeeded")


if __name__ == "__main__":
    main()
