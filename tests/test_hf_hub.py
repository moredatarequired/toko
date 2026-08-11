"""Tests for the Hub rate-limit helper.

The transport is stubbed because a 429 cannot be summoned on demand; everything above it
is the real `requests` stack and the real helper.
"""

import io

import pytest
import requests
import requests.adapters
import urllib3

from tests.hf_hub import STRICT_ENV_VAR, skip_if_rate_limited

RESOLVE_URL = (
    "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/tokenizer.json"
)
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"


def _stub_transport(monkeypatch, statuses: list[int]) -> None:
    remaining = list(statuses)

    def send(_self, request, *_args, **_kwargs):
        status = remaining.pop(0) if remaining else 200
        response = requests.Response()
        response.status_code = status
        response.url = str(request.url)
        response.request = request
        response.raw = urllib3.HTTPResponse(
            body=io.BytesIO(b"{}"), status=status, preload_content=False
        )
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)


def _fetch(url: str = RESOLVE_URL) -> None:
    requests.get(url, timeout=5)


def _load_refused_by_the_hub() -> None:
    with skip_if_rate_limited():
        _fetch()
        raise ValueError("Failed to count tokens: 429 Client Error")


def _block_asserting_on_the_result() -> None:
    with skip_if_rate_limited():
        _fetch()
        count = 1
        assert count == 99999, "real failure"


def _block_reporting_collected_failures() -> None:
    with skip_if_rate_limited():
        _fetch()
        pytest.fail("models failed to count tokens:\ngpt-5 (openai): boom")


def _block_failing_for_another_reason() -> None:
    with skip_if_rate_limited():
        _fetch()
        raise ValueError("server error")


def _block_hitting_an_unrelated_endpoint() -> None:
    with skip_if_rate_limited():
        _fetch(WHOAMI_URL)
        raise ValueError("unrelated")


def _block_that_swallows_the_refusal() -> None:
    with skip_if_rate_limited():
        _fetch()


def test_rate_limited_load_skips(monkeypatch):
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.skip.Exception, match="rate limit"):
        _load_refused_by_the_hub()


def test_assertion_failure_beats_the_skip(monkeypatch):
    _stub_transport(monkeypatch, [429])
    with pytest.raises(AssertionError, match="real failure"):
        _block_asserting_on_the_result()


def test_reported_failure_beats_the_skip(monkeypatch):
    """A 429 must not discard failures collected for providers that never touch the Hub.

    This is the `test_every_listed_model_counts_tokens` shape.
    """
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.fail.Exception, match="gpt-5"):
        _block_reporting_collected_failures()


def test_failure_without_a_rate_limit_is_not_skipped(monkeypatch):
    _stub_transport(monkeypatch, [500])
    with pytest.raises(ValueError, match="server error"):
        _block_failing_for_another_reason()


def test_recovered_rate_limit_does_not_skip(monkeypatch):
    """A 429 the Hub later answered is not a reason to throw the run away.

    `huggingface_hub` retries 429s on its paginated endpoints.
    """
    _stub_transport(monkeypatch, [429, 200])
    with skip_if_rate_limited():
        _fetch()
        _fetch()


def test_rate_limit_on_an_unrelated_path_does_not_skip(monkeypatch):
    _stub_transport(monkeypatch, [429])
    with pytest.raises(ValueError, match="unrelated"):
        _block_hitting_an_unrelated_endpoint()


def test_unanswered_rate_limit_skips_a_block_that_did_not_raise(monkeypatch):
    """The CLI swallows the refusal, so the assertions that catch it come after the block.

    This is the `test_partial_success_missing_hf_token` shape.
    """
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.skip.Exception, match="rate limit"):
        _block_that_swallows_the_refusal()


def test_strict_mode_turns_the_skip_into_a_failure(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.fail.Exception, match="rate limit"):
        _load_refused_by_the_hub()
