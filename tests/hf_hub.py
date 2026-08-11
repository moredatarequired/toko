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

import pytest
import requests.adapters

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


@contextlib.contextmanager
def skip_if_rate_limited():
    """Skip the test if the Hub refused a tokenizer fetch with 429 and never relented.

    Detection is on the HTTP status rather than on an error message, so an unreachable
    Hub, a missing model and a broken tokenizer all still fail the test.

    A verdict the block reached itself — `assert`, `pytest.fail` — always wins over the
    skip. Otherwise a single Hub 429 could discard a whole test's findings, including
    findings about providers that never touch the Hub.

    The skip still fires for a block that completed without raising, because a caller can
    swallow the refusal and surface it as changed output rather than as an exception:
    `test_partial_success_missing_hf_token` gets exit code 0 and a warning naming a
    connection error instead of `HF_TOKEN`, and only the assertions after the block can
    see it. That is also why a 429 the Hub never answered is treated as fatal to the
    block even when a warm `HF_HOME` would have served the file from cache — a refusal
    that the cache silently covered is indistinguishable here from one that broke the
    run, and skipping a working test costs less than failing a working one.
    """
    original_send = requests.adapters.HTTPAdapter.send
    refused: dict[str, None] = {}

    def send(self, request, *args, **kwargs):
        response = original_send(self, request, *args, **kwargs)
        url = str(request.url)
        if _is_tokenizer_fetch(url):
            if response.status_code == 429:
                refused[url] = None
            else:
                # The Hub answered this URL after all, whether on a retry or a later
                # request, so it was never what stood between the test and its data.
                refused.pop(url, None)
        return response

    patch = pytest.MonkeyPatch()
    patch.setattr(requests.adapters.HTTPAdapter, "send", send)
    try:
        try:
            yield
        finally:
            patch.undo()
    except VERDICT_EXCEPTIONS:
        raise
    except Exception:
        if not refused:
            raise

    if refused:
        message = f"Hugging Face Hub rate limit (429) on {next(iter(refused))}"
        if os.environ.get(STRICT_ENV_VAR):
            pytest.fail(message)
        pytest.skip(message)
