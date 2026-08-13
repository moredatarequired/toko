"""The release smoke test's own checks, run at PR time.

`tests/smoke_test.py` is only invoked by `.github/workflows/release.yml`, against an
installed artifact, after a tag is pushed. Anything that first runs there fails when
publishing is already underway, so everything in it that does not need an installed
artifact is exercised here instead.
"""

import os
import re
from pathlib import Path

import pytest

from tests import smoke_test


def test_optional_groups_still_declare_the_models_the_smoke_test_names():
    """Both names are pinned in the smoke test, so a registry edit must not orphan them."""
    assert smoke_test.MISTRAL_MODEL in smoke_test.optional_models("mistral_common")
    assert smoke_test.TRANSFORMERS_MODEL in smoke_test.optional_models("transformers")

    with pytest.raises(AssertionError, match="No optional group"):
        smoke_test.optional_models("nonesuch")


def test_the_provider_listing_matches_what_the_smoke_test_expects():
    """The exact-set assertion, where adding a provider costs a red PR, not a release."""
    smoke_test.check_listing()


def test_the_mistral_count_is_plausible():
    smoke_test.check_mistral()


# Every string below is a verbatim `str(ValueError)` out of `_count_transformers`,
# captured from a failure induced against transformers 4.57.3 / huggingface_hub 0.36.0.
# Where inducing it meant choosing the Hub's answer, HF_ENDPOINT pointed at a local stand-
# in, so the host in that message is 127.0.0.1 rather than huggingface.co.
HUB_OUTAGE_MESSAGES = {
    # The HEAD call failed. Captured under HF_HUB_OFFLINE=1 with a cold cache; an endpoint
    # answering 429, 500 or 503, a refused connection, an unresolvable host and a connect
    # timeout produce the same text, because hf_hub_download converts every one of them to
    # a LocalEntryNotFoundError and the status code never reaches here.
    "the head call failed": (
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't "
        "connect to 'https://huggingface.co' to load the files, and couldn't find them "
        "in the cached files.\nCheck your internet connection or see how to run the "
        "library in offline mode at "
        "'https://huggingface.co/docs/transformers/installation#offline-mode'."
    ),
    # The HEAD call succeeded and the GET failed, so the HfHubHTTPError reached
    # transformers intact. The only shape that carries the status code.
    "the get call failed": (
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: There was a "
        "specific connection error when trying to load Qwen/Qwen2.5-7B-Instruct:\n"
        "503 Server Error: Service Unavailable for url: "
        "http://127.0.0.1:19503/api/resolve-cache/models/Qwen/Qwen2.5-7B-Instruct/"
        "a09a35458c702b33eeacc393d103063234e8bc28/config.json"
    ),
}

# huggingface_hub re-raises SSLError and ProxyError ahead of the LocalEntryNotFoundError
# conversion, so a runner that cannot make an HTTPS request arrives with its own text
# rather than the Hub's. Excusing any of these publishes a release that never reached the
# Hub — and a "connection" marker did excuse all three, by way of "HTTPSConnectionPool".
BROKEN_RUNNER_MESSAGES = {
    "the ca bundle is not a certificate": (
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: "
        "(MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', port=443): Max "
        "retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/"
        "tokenizer_config.json (Caused by SSLError(SSLError(136, '[X509: "
        "NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found (_ssl.c:4416)')))\"), "
        "'(Request ID: 01d7c1dc-5fda-4b5b-a871-1b9310dc5fdb)')"
    ),
    "the endpoint certificate does not verify": (
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: "
        "(MaxRetryError(\"HTTPSConnectionPool(host='127.0.0.1', port=18443): Max retries "
        "exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json "
        "(Caused by SSLError(SSLCertVerificationError(1, '[SSL: "
        "CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate "
        "(_ssl.c:1082)')))\"), '(Request ID: 2f84771e-414e-4f60-8aeb-c022ec4718f0)')"
    ),
    "the proxy is not listening": (
        "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: "
        "(MaxRetryError('HTTPSConnectionPool(host=\\'huggingface.co\\', port=443): Max "
        "retries exceeded with url: /Qwen/Qwen2.5-7B-Instruct/resolve/main/"
        "tokenizer_config.json (Caused by ProxyError(\\'Unable to connect to proxy\\', "
        "NewConnectionError(\"HTTPSConnection(host=\\'127.0.0.1\\', port=9): Failed to "
        "establish a new connection: [Errno 111] Connection refused\")))'), '(Request "
        "ID: 6ddf7d2c-1261-4388-b64a-47a2efa243f6)')"
    ),
}


@pytest.mark.parametrize(
    "reason", HUB_OUTAGE_MESSAGES.values(), ids=HUB_OUTAGE_MESSAGES
)
def test_a_hub_outage_is_recognised(reason):
    assert smoke_test.hub_was_unavailable(reason)


@pytest.mark.parametrize(
    "reason", BROKEN_RUNNER_MESSAGES.values(), ids=BROKEN_RUNNER_MESSAGES
)
def test_a_broken_runner_is_not_excused_as_a_hub_outage(reason):
    assert not smoke_test.hub_was_unavailable(reason)


@pytest.mark.parametrize("marker", smoke_test.HUB_UNAVAILABLE_MARKERS)
def test_every_marker_earns_its_place(marker, monkeypatch):
    """Each marker is a way a broken release passes, so each must be load-bearing here."""
    monkeypatch.setattr(
        smoke_test,
        "HUB_UNAVAILABLE_MARKERS",
        tuple(m for m in smoke_test.HUB_UNAVAILABLE_MARKERS if m != marker),
    )
    assert not all(map(smoke_test.hub_was_unavailable, HUB_OUTAGE_MESSAGES.values()))


def test_the_status_pattern_earns_its_place(monkeypatch):
    """The other half of the guard, and the only half that reads a status code."""
    monkeypatch.setattr(smoke_test, "_HUB_STATUS", re.compile(r"(?!)"))
    assert not all(map(smoke_test.hub_was_unavailable, HUB_OUTAGE_MESSAGES.values()))


@pytest.mark.parametrize(
    "reason",
    [
        # The status match is anchored, so digits inside a name or an offset are not it.
        "Model 'org/model-429b' is gated on Hugging Face. Accept the license",
        "Model 'Qwen/Qwen2.5-72B' requires authentication. Set HF_TOKEN",
        "Failed to count tokens for Qwen model X: shape mismatch at offset 5043",
        # A bare status pattern matched all of these, so a broken tokenizer could print
        # "skipped" and let the release publish.
        "Failed to count tokens for Qwen model X: expected 512 tokens, got 7",
        "Failed to count tokens for Qwen model X: bad merges table (see line 501)",
        "Failed to count tokens for Qwen model X: vocab size 500 does not match 4096",
    ],
)
def test_a_failure_the_hub_did_not_cause_is_not_excused(reason):
    assert not smoke_test.hub_was_unavailable(reason)


def test_the_isolation_restores_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    with smoke_test.isolated_cache_home():
        # setdefault, so a caller that already said where keeps its answer.
        assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)
        created = Path(os.environ["XDG_CONFIG_HOME"])
        assert created != tmp_path

    assert os.environ["XDG_CACHE_HOME"] == str(tmp_path)
    assert "XDG_CONFIG_HOME" not in os.environ
    assert not created.exists()


@pytest.mark.slow
def test_the_whole_smoke_test_passes():
    """The release invocation itself, minus the installed artifact."""
    smoke_test.main()
