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
import importlib
import importlib.util
import io
import os
import shutil
import ssl
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING

import toko
from toko.cli import app
from toko.counter import count_tokens
from toko.models import OPTIONAL_GROUPS, list_models

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

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
# on its own schedule. An outage there is not a reason to block a release, so it prints
# "skipped" — but a gated repo, a missing model, a broken tokenizer, or a runner that
# cannot make an HTTPS request at all is a real failure and still fails the release.
#
# The guard reads exception classes, not message text, because the release does not run
# the libraries this repository is locked to. The dev lockfile pins huggingface_hub 0.36
# on requests with transformers 4.57; the release smoke steps install unlocked and
# resolve huggingface_hub 1.27 on httpx with transformers 5.15. The two word the same
# outage differently — 0.36 says "429 Client Error", 1.27 says "429 Too Many Requests",
# and httpx underneath it says "Client error '429 Too Many Requests'" — so any wording
# pinned from one environment is wrong in the other. The classes are identical in both.
#
# Reading them is possible because nothing in the stack flattens the chain:
# `_count_transformers` funnels every failure into a ValueError but raises it `from` the
# original, every branch of transformers' `utils/hub.py` re-raises `from e`, and
# huggingface_hub's `_raise_on_head_call_error` raises
# `LocalEntryNotFoundError(...) from head_call_error`. Both halves of an outage survive:
#
#   - The HEAD call fails. `_raise_on_head_call_error` converts 429, every 5xx, a refused
#     connection, a failed DNS lookup, a timeout and HF_HUB_OFFLINE=1 alike into
#     `LocalEntryNotFoundError`, re-raising 401, gated and repo-not-found unconverted
#     instead. So the class alone means "the Hub could not be reached".
#   - The HEAD call succeeds and the GET fails. Nothing converts that, so the HTTP error
#     arrives intact and carries the status code the Hub answered with.
#
# One shape is not distinguishable here, and is deliberately left excused rather than
# guessed at. A proxy that refuses the TCP connection reaches us as
# `LocalEntryNotFoundError` -> `httpx.ConnectError` -> `ConnectionRefusedError`, which is
# link for link what the Hub itself refusing the connection produces; nothing in the
# chain names the proxy, and nothing in the message does either. So on a release runner
# whose proxy is down, this prints "skipped" and the release publishes. Reading the
# environment's own proxy variables to tell the two apart would be a guess about which
# host the refusal came from, so the guard does not: a runner misconfigured that way is
# outside what a smoke test can catch. Only the release stack has this gap — under
# requests, huggingface_hub 0.36 raises `ProxyError` instead of converting, so no outage
# rule ever reaches it and it fails the release on its own.

# 429 is the Hub rate-limiting an anonymous caller and 5xx is the Hub failing; both are
# the Hub rather than the release. The whole 5xx range rather than the 500 and 503 that
# were induced, because "the server failed" is what the class of code means and the Hub
# sits behind a CDN that has its own; picking out individual codes would draw a line no
# measurement supports. 401, 403 and 404 are deliberately absent, and are the reason this
# is a set of codes rather than "any status at all": those are a bad token, a gated repo
# and a wrong model name, and each has to fail the release.
HUB_OUTAGE_STATUS = frozenset({429, *range(500, 600)})

# The longest chain measured is 11 links — a count that tried two files before giving up,
# the second failure chaining the first through `__context__` — so the cap is a runaway
# guard rather than a policy, set well clear of a load that reaches for more files. It has
# to be generous in that direction: the `ssl.SSLError` that denies a broken runner sat at
# link 6 of 6, and a cap that stopped short of it would excuse a release that never
# reached the Hub. `_causes` therefore refuses to judge a chain it could not read to the
# end rather than ruling on a partial view.
_CHAIN_LIMIT = 200


def _causes(error: BaseException) -> list[BaseException]:
    """Walk from `error` through everything it was raised from, outermost first.

    `__context__` can point back into a chain already walked, so the walk is both
    cycle-safe and bounded. Overrunning the bound returns nothing, which fails the
    release: an unread chain may hold the SSL cause `BROKEN_RUNNER_TYPES` looks for.
    """
    links: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if len(links) == _CHAIN_LIMIT:
            return []
        seen.add(id(current))
        links.append(current)
        current = current.__cause__ or current.__context__
    return links


@lru_cache(maxsize=1)
def _hub_errors() -> ModuleType:
    """huggingface_hub's error classes.

    Imported lazily: the no-extras release step installs neither transformers nor
    huggingface_hub, and never reaches a call that could need them.
    """
    return importlib.import_module("huggingface_hub.errors")


def _hub_reported_downtime(link: BaseException) -> bool:
    """huggingface_hub's own verdict that it could not reach the Hub and had no cache."""
    return isinstance(link, _hub_errors().LocalEntryNotFoundError)


def _hub_returned_an_outage_status(link: BaseException) -> bool:
    """Report a status the Hub answered with, off whichever exception carries it.

    Deliberately not tied to `HfHubHTTPError`: huggingface_hub wraps the transport's
    error in its own class on some paths and lets requests' or httpx's through on others,
    and which one arrives differs between the two resolved stacks — a GET refused with
    503 arrives as `HfHubHTTPError` under 1.27 but as a bare
    `requests.exceptions.HTTPError` under 0.36. Every request in this code path goes to
    the Hub, so a status is the Hub's answer whoever is holding it.
    """
    response = getattr(link, "response", None)
    return getattr(response, "status_code", None) in HUB_OUTAGE_STATUS


HUB_OUTAGE_RULES = (_hub_reported_downtime, _hub_returned_an_outage_status)


# The runner's own TLS being broken is the one failure that an outage rule would
# otherwise excuse, so it is denied ahead of them. huggingface_hub 1.x re-raises only
# `httpx.ProxyError` before the downtime conversion and lets a certificate failure through
# as an `httpx.ConnectError`, so a self-signed endpoint arrives wearing
# `LocalEntryNotFoundError` and reads as "We couldn't connect" — excused by the outage
# rules, and by any wording a message could be matched on. Walking to the end of the chain
# is what finds the `ssl.SSLError` under it.
#
# One class, because one class is all any measured failure needs. It covers both stacks:
# under requests the chain runs `requests.exceptions.SSLError` -> urllib3 -> `ssl.SSLError`
# anyway. No proxy class is named here — a proxy failure is never converted into
# `LocalEntryNotFoundError` by either version and carries no status, so no outage rule
# reaches it and nothing has to hold it back.
BROKEN_RUNNER_TYPES: tuple[type[BaseException], ...] = (ssl.SSLError,)


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


def hub_was_unavailable(error: BaseException) -> bool:
    """Whether `error` is the Hub being down rather than anything about this release."""
    links = _causes(error)
    # Denial first: huggingface_hub 1.x converts a certificate failure into the same
    # class it uses for genuine downtime, so the outage rules would otherwise excuse it.
    if any(isinstance(link, BROKEN_RUNNER_TYPES) for link in links):
        return False
    return any(rule(link) for link in links for rule in HUB_OUTAGE_RULES)


def check_transformers() -> None:
    if importlib.util.find_spec("transformers") is None:
        print("transformers: skipped: transformers is not installed")
        return
    assert TRANSFORMERS_MODEL in optional_models("transformers")
    try:
        counted = count_tokens(SAMPLE, model=TRANSFORMERS_MODEL, use_cache=False)
    except ValueError as error:
        if not hub_was_unavailable(error):
            raise
        print(f"transformers ({TRANSFORMERS_MODEL}): skipped: {error}")
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
