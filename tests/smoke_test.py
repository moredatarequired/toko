"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work.

Run it as `python tests/smoke_test.py`, which is how the release workflow invokes it
against an installed artifact. Importing it runs no check, creates no directory and
redirects no cache — which matters because the file matches pytest's default
`*_test.py` glob, so an ordinary collection imports it. `tests/test_release_smoke.py`
exercises the checks below at PR time, where a failure costs a red build rather than a
blocked release.
"""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import tempfile
from typing import TYPE_CHECKING

import toko
from toko.cli import app
from toko.counter import count_tokens
from toko.models import OPTIONAL_GROUPS, list_models

if TYPE_CHECKING:
    from collections.abc import Iterator

    from toko.result import TokenCount

SAMPLE = "The most basic of smoke tests."

# Named rather than indexed out of OPTIONAL_GROUPS: the transformers check downloads
# this tokenizer, and reordering TRANSFORMERS_MODELS must not silently retarget it at a
# gated repository, whose "requires authentication" is not a failure to tolerate.
MISTRAL_MODEL = "mistral-small-latest"
TRANSFORMERS_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CORE_PROVIDERS = {"anthropic", "google", "openai", "xai"}
# The only optional gate in `list_models` is `_has_module("transformers")`; installing
# mistral-common changes the listing not at all, so no `mistral` provider ever appears
# and the Mistral model name has to come from `OPTIONAL_GROUPS`.
TRANSFORMERS_PROVIDERS = {"deepseek", "huggingface", "llama", "qwen"}

# A tokenizer load reaches the Hugging Face Hub, which refuses anonymous callers with 429
# on its own schedule. `_count_transformers` funnels every failure into a ValueError, so
# what the Hub did is only visible in the message; anything these do not cover — a gated
# repo, a missing model, a broken tokenizer — is a real failure and still fails the
# release.
#
# The status code is almost never in that message. `hf_hub_download` and
# `snapshot_download` both convert a 429 or a 5xx HEAD failure into a bare
# `LocalEntryNotFoundError` (only 401, gated and repo-not-found keep their
# `HfHubHTTPError`), and transformers then reports that as "We couldn't connect to ...".
# Against a live endpoint returning 429, 500 or 503 alike, and under HF_HUB_OFFLINE=1
# with a cold cache, the whole of what arrives here is:
#
#     Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't
#     connect to 'https://huggingface.co' to load the files, and couldn't find them in
#     the cached files.
#     Check your internet connection or see how to run the library in offline mode at
#     'https://huggingface.co/docs/transformers/installation#offline-mode'.
#
# So the markers, not the status pattern, are what actually excuses an outage. The
# pattern stays for the paths that do pass an `HfHubHTTPError` through verbatim, and is
# anchored on the phrase requests puts after the code, because a bare `\b5\d\d\b` also
# matches "expected 512 tokens, got 7" — turning a genuinely broken tokenizer into a
# printed skip, which is the one thing this guard must never do.
_HUB_STATUS = re.compile(r"\b(?:429|5\d\d) (?:client|server) error\b")
HUB_UNAVAILABLE_MARKERS = (
    # Both the "local cache" and the "disk cache" wording of LocalEntryNotFoundError.
    "cannot find the requested files",
    "connection",
    "couldn't connect",
    "name resolution",
    "rate limit",
    "timed out",
    "timeout",
    "too many requests",
)


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


def optional_models(module: str) -> tuple[str, ...]:
    for group in OPTIONAL_GROUPS:
        if group.module == module:
            return group.models
    raise AssertionError(f"No optional group declares module {module!r}")


def _assert_plausible(counted: TokenCount) -> None:
    # A band rather than a number: SAMPLE is 7 tokens to tiktoken, and a tokenizer whose
    # revision or chat framing shifts that by a few is not a broken tokenizer. One that
    # lands outside this is not counting SAMPLE at all.
    assert 5 <= counted.count <= 20, counted


def check_core() -> None:
    assert toko.__version__ is not None
    assert toko.__version__ != "0.0.0"
    counted = count_tokens(SAMPLE, model="gpt-5")
    assert counted.count == 7
    print(f"tiktoken (gpt-5): {counted.count} tokens")


def check_listing() -> None:
    models_by_provider = list_models()
    providers = set(models_by_provider)
    expected = CORE_PROVIDERS
    if importlib.util.find_spec("transformers") is not None:
        expected = CORE_PROVIDERS | TRANSFORMERS_PROVIDERS
    assert providers == expected, providers
    assert len(models_by_provider["anthropic"]) > 5
    assert len(models_by_provider["openai"]) > 5
    print(f"providers listed: {', '.join(sorted(providers))}")


def check_mistral() -> None:
    if importlib.util.find_spec("mistral_common") is None:
        print("mistral: skipped: mistral_common is not installed")
        return
    assert MISTRAL_MODEL in optional_models("mistral_common")
    counted = count_tokens(SAMPLE, model=MISTRAL_MODEL, use_cache=False)
    _assert_plausible(counted)
    print(f"mistral ({MISTRAL_MODEL}): {counted.count} tokens")


def hub_was_unavailable(reason: str) -> bool:
    lowered = reason.lower()
    if _HUB_STATUS.search(lowered):
        return True
    return any(marker in lowered for marker in HUB_UNAVAILABLE_MARKERS)


def check_transformers() -> None:
    if importlib.util.find_spec("transformers") is None:
        print("transformers: skipped: transformers is not installed")
        return
    assert TRANSFORMERS_MODEL in optional_models("transformers")
    try:
        counted = count_tokens(SAMPLE, model=TRANSFORMERS_MODEL, use_cache=False)
    except ValueError as error:
        reason = str(error)
        if not hub_was_unavailable(reason):
            raise
        print(f"transformers ({TRANSFORMERS_MODEL}): skipped: {reason}")
        return
    _assert_plausible(counted)
    print(f"transformers ({TRANSFORMERS_MODEL}): {counted.count} tokens")


def check_cli() -> None:
    # Running the CLI, not just importing it, is what catches a dependency that toko
    # imports but never declares — the failure mode typer 0.26 exposed by dropping click.
    # It only catches it where the extras do not supply the module anyway, which is why
    # the release runs this against a no-extras install as well as against [all].
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
    print("cli: --version and separated-option dispatch both reached")


def main() -> None:
    with isolated_cache_home():
        check_core()
        check_listing()
        check_mistral()
        check_transformers()
        check_cli()
    print("Smoke test succeeded")


if __name__ == "__main__":
    main()
