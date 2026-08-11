"""Test support for reaching the Hugging Face Hub, which rate-limits anonymous callers.

`transformers.AutoTokenizer.from_pretrained` asks the Hub for model metadata on every
load — `_patch_mistral_regex` calls `model_info()`, whose response is never cached — so
even a fully warm `HF_HOME` still needs the Hub to answer. Anonymous requests share a
per-IP quota, so a run can be refused with HTTP 429 while the code under test is fine.

`requests` is not a declared toko dependency; it arrives transitively via `tiktoken` and
`huggingface-hub`, which is why patching its adapter reaches the Hub traffic at all.
`huggingface_hub` 1.x switches to `httpx`, at which point this interceptor sees nothing
and every 429 becomes a plain test failure again. That is the safe direction to fail —
loudly, rather than silently green — but it means this module needs revisiting on that
upgrade rather than being quietly trusted.
"""

import contextlib
import os
import urllib.parse
from typing import TYPE_CHECKING

import pytest
import requests.adapters

if TYPE_CHECKING:
    from collections.abc import Iterator

HUB_HOST = "huggingface.co"

# The Hub paths a tokenizer load actually goes through: file downloads (`/resolve/`),
# model metadata and the repo tree (`/api/models...`), and the redirect target the Hub
# sends `/resolve/` HEADs to (`/api/resolve-cache/...`). A 429 anywhere else on the Hub —
# `/api/whoami-v2`, say — says nothing about whether the code under test can be reached.
FETCH_PATH_PREFIXES = ("/api/models", "/api/resolve-cache")

# Set in a scheduled run to turn these skips back into failures, so a Hub outage that
# lasts for days shows up as a red build instead of a green one that checked nothing.
STRICT_ENV_VAR = "TOKO_HF_STRICT"

# A test that reached its own verdict — assert, pytest.fail, pytest.skip — is reporting
# on the code under test, not on the Hub, so it must never be overridden by a skip.
VERDICT_EXCEPTIONS = (AssertionError, pytest.fail.Exception, pytest.skip.Exception)


def _is_tokenizer_fetch(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != HUB_HOST:
        return False
    return "/resolve/" in parts.path or parts.path.startswith(FETCH_PATH_PREFIXES)


class HubTraffic:
    """What the Hub did to tokenizer fetches while the guarded block ran.

    `refused` is the question the skip turns on: some URL was refused and the Hub never
    went on to answer it. `refusals_seen` is the wider count, incremented by every 429
    including ones a retry got past, so a block that catches its own errors can tell
    which of them the Hub caused — snapshot it either side of each operation.

    It also carries `masked`, because it is the one object both `skip_if_rate_limited`
    and a collecting block hold: see `mask`.
    """

    def __init__(self) -> None:
        self.refusals_seen = 0
        self.masked: list[Exception] = []
        self._unanswered: dict[str, None] = {}

    @property
    def refused(self) -> bool:
        return bool(self._unanswered)

    @property
    def first_refused_url(self) -> str:
        return next(iter(self._unanswered))

    def record(self, url: str, status_code: int) -> None:
        if status_code == 429:
            self.refusals_seen += 1
            self._unanswered[url] = None
        else:
            # The Hub answered this URL after all, whether on a retry or a later
            # request, so it was never what stood between the test and its data.
            self._unanswered.pop(url, None)

    def mask(self, errors: list[Exception]) -> None:
        """Take custody of errors a block chose not to report because the Hub caused them.

        Handing them here rather than dropping them is what lets `skip_if_rate_limited`
        name them in the skip reason and chain them onto it, so `-rs` shows what was
        hidden and strict mode fails carrying the original errors.
        """
        self.masked.extend(errors)


class HubFailures:
    """Failures a guarded block collects instead of raising, split by who is to blame.

    A block that loops over many models and gathers one message per broken one cannot
    just let `skip_if_rate_limited` sort things out: it never raises the Hub's error, it
    reports a `pytest.fail` built out of it, and a `pytest.fail` is a verdict the block
    reached itself, so it beats the skip every time. Attribution has to happen where the
    errors are caught, which is here.
    """

    def __init__(self, hub: HubTraffic) -> None:
        self._hub = hub
        self.genuine: list[str] = []
        self.rate_limited: list[str] = []
        self._rate_limited_errors: list[Exception] = []

    @contextlib.contextmanager
    def collect(self, label: str) -> Iterator[None]:
        refusals_before = self._hub.refusals_seen
        try:
            yield
        except VERDICT_EXCEPTIONS:
            # Same rule as the outer guard: a verdict the block reached itself is about
            # the code under test, so collecting it — and then dropping it as the Hub's
            # doing — would lose it entirely. `pytest.fail.Exception` and
            # `pytest.skip.Exception` derive from `BaseException` and never reach the
            # clause below anyway; `AssertionError` does, which is why this comes first.
            raise
        except Exception as exc:
            entry = f"{label}: {exc}"
            if self._hub.refusals_seen > refusals_before:
                self.rate_limited.append(entry)
                self._rate_limited_errors.append(exc)
            else:
                self.genuine.append(entry)

    def report(self, headline: str) -> None:
        """Fail with the failures the Hub cannot account for; stay silent if it can.

        Rate-limited entries are dropped only when `skip_if_rate_limited` is going to
        skip on the way out. If every 429 was answered on a retry there is no skip
        coming, and nothing is left excusing those failures, so they are reported —
        alongside any genuine ones rather than instead of them, since a real regression
        is no reason to stop reporting a failure the Hub has stopped accounting for.

        Dropping is never discarding: the errors behind the dropped entries go to the
        `HubTraffic` record, which is how they reach the skip reason and strict mode.
        """
        reported = self.genuine + ([] if self._hub.refused else self.rate_limited)
        if not reported:
            self._hub.mask(self._rate_limited_errors)
            return
        formatted = "\n".join(reported)
        omitted = (
            f"\n(plus {len(self.rate_limited)} refused by the Hub with 429, not listed)"
            if self._hub.refused and self.rate_limited
            else ""
        )
        pytest.fail(f"{headline}\n{formatted}{omitted}")


def _chained(errors: list[Exception]) -> BaseException | None:
    """Build the cause to hang the skip off, keeping every masked error's traceback.

    A sweep the Hub refuses masks one error per model — eight in an observed run — but
    `raise ... from` takes a single cause, so chaining `errors[0]` would name all eight
    in the reason and throw seven tracebacks away, in the one mode that exists to
    diagnose them. A group of one is noise, so a lone error is still chained directly.
    """
    if not errors:
        return None
    if len(errors) == 1:
        return errors[0]
    return ExceptionGroup(f"{len(errors)} errors masked by the rate limit", errors)


@contextlib.contextmanager
def skip_if_rate_limited() -> Iterator[HubTraffic]:
    """Skip the test if the Hub refused a tokenizer fetch with 429 and never relented.

    Detection is on the HTTP status rather than on an error message, so an unreachable
    Hub, a missing model and a broken tokenizer all still fail the test.

    A verdict the block reached itself — `assert`, `pytest.fail` — always wins over the
    skip. Otherwise a single Hub 429 could discard a whole test's findings, including
    findings about providers that never touch the Hub. A block that collects failures
    rather than raising them needs `HubFailures` to keep that guarantee.

    The skip still fires for a block that completed without raising, because a caller can
    swallow the refusal and surface it as changed output rather than as an exception:
    `test_partial_success_missing_hf_token` gets exit code 0 and a warning naming a
    connection error instead of `HF_TOKEN`, and only the assertions after the block can
    see it. That is also why a 429 the Hub never answered is treated as fatal to the
    block even when a warm `HF_HOME` would have served the file from cache — a refusal
    that the cache silently covered is indistinguishable here from one that broke the
    run, and skipping a working test costs less than failing a working one.

    Yields the `HubTraffic` record so the block can attribute its own errors.
    """
    original_send = requests.adapters.HTTPAdapter.send
    hub = HubTraffic()

    def send(self, request, *args, **kwargs):
        response = original_send(self, request, *args, **kwargs)
        url = str(request.url)
        if _is_tokenizer_fetch(url):
            hub.record(url, response.status_code)
        return response

    patch = pytest.MonkeyPatch()
    patch.setattr(requests.adapters.HTTPAdapter, "send", send)
    swallowed: list[Exception] = []
    try:
        try:
            yield hub
        finally:
            patch.undo()
    except VERDICT_EXCEPTIONS:
        raise
    except Exception as exc:
        if not hub.refused:
            raise
        # A refusal reaches the caller as an ordinary exception ("Failed to count tokens
        # for Qwen model ...: 429 Client Error"), and this module classifies on HTTP
        # status rather than on message text, so a genuine non-assertion regression that
        # happened to coincide with a 429 is indistinguishable from it here. Swallowing
        # it is the same trade as above: under a Hub outage, skipping a test that works
        # costs less than failing one that works. It is not discarded, though — it is
        # chained onto the skip below and named in the reason, so `-rs` still shows it
        # and strict mode still fails on it.
        swallowed.append(exc)

    # A block that collects errors rather than raising them makes the same trade one
    # model at a time, and hands what it dropped to the same place, so the guarantee
    # above holds however the block is written.
    swallowed.extend(hub.masked)

    if hub.refused:
        message = f"Hugging Face Hub rate limit (429) on {hub.first_refused_url}"
        for exc in swallowed:
            message += f"\nThe guarded block also raised {type(exc).__name__}: {exc}"
        cause = _chained(swallowed)
        if os.environ.get(STRICT_ENV_VAR):
            raise pytest.fail.Exception(message) from cause
        raise pytest.skip.Exception(message) from cause
