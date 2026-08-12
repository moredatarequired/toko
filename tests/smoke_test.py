"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work.
"""

import contextlib
import io

import toko
from toko.cli import app
from toko.counter import count_tokens
from toko.models import list_models

assert toko.__version__ is not None
assert toko.__version__ != "0.0.0"
assert count_tokens("The most basic of smoke tests.", model="gpt-5").count == 7

models_by_provider = list_models()
assert len(models_by_provider) > 4
assert len(models_by_provider["anthropic"]) > 5
assert len(models_by_provider["openai"]) > 5

# Running the CLI, not just importing it, is what catches a dependency that toko
# imports but never declares — the failure mode typer 0.26 exposed by dropping click.
version_output = io.StringIO()
with contextlib.redirect_stdout(version_output), contextlib.suppress(SystemExit):
    app(["--version"])
assert toko.__version__ in version_output.getvalue()

# --version is an eager no-value flag, so it never reaches the option scan TokoGroup
# uses to find a subcommand past a global option. Dispatch a subcommand behind a
# separated option value instead: if a future typer changes how params are built, the
# scan stops recognising -m as value-taking and "clear-cache" is read as a path.
dispatch_output = io.StringIO()
with contextlib.redirect_stdout(dispatch_output), contextlib.suppress(SystemExit):
    app(["-m", "gpt-5", "clear-cache"])
assert "Cache cleared" in dispatch_output.getvalue()

print("Smoke test succeeded")
