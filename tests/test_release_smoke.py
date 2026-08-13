"""The release smoke test's own checks, run at PR time.

`tests/smoke_test.py` is only invoked by `.github/workflows/release.yml`, against an
installed artifact, after a tag is pushed. Anything that first runs there fails when
publishing is already underway, so everything in it that does not need an installed
artifact is exercised here instead.
"""

import os
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


# What `_count_transformers` actually raises when the Hub is unreachable, captured
# verbatim from a run against an endpoint answering 429 (500, 503 and HF_HUB_OFFLINE=1
# with a cold cache all produce the identical text — the status code never reaches here,
# because hf_hub_download converts every one of them to LocalEntryNotFoundError first).
HUB_OUTAGE_MESSAGE = (
    "Failed to count tokens for Qwen model Qwen/Qwen2.5-7B-Instruct: We couldn't "
    "connect to 'https://huggingface.co' to load the files, and couldn't find them in "
    "the cached files.\nCheck your internet connection or see how to run the library in "
    "offline mode at "
    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
)


@pytest.mark.parametrize(
    "reason",
    [
        HUB_OUTAGE_MESSAGE,
        "429 Client Error: Too Many Requests for url: https://huggingface.co/...",
        "We couldn't connect to 'https://huggingface.co' to load this file",
        "503 Server Error: Service Unavailable",
        "HTTPSConnectionPool: Read timed out.",
        # The one shape that does carry the code: transformers reporting an HfHubHTTPError
        # it was handed rather than a LocalEntryNotFoundError.
        "There was a specific connection error when trying to load Qwen/Qwen2.5-7B-"
        "Instruct:\n429 Client Error: Too Many Requests for url: https://huggingface.co/",
    ],
)
def test_a_hub_outage_is_recognised(reason):
    assert smoke_test.hub_was_unavailable(reason)


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
