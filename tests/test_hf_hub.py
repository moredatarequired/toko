"""Tests for the Hub rate-limit helper.

The transport is stubbed because a 429 cannot be summoned on demand; everything above it
is the real `requests` stack and the real helper.
"""

import io

import pytest
import requests
import requests.adapters
import urllib3

from tests.hf_hub import STRICT_ENV_VAR, HubFailures, skip_if_rate_limited

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


def _refused_by_the_hub() -> None:
    _fetch()
    raise RuntimeError("Failed to count tokens for Qwen/Qwen2-7B: 429 Client Error")


def _refused_then_answered() -> None:
    _fetch()
    _fetch()
    raise RuntimeError("Failed to count tokens for Qwen/Qwen2-7B: 429 Client Error")


def _broken_tokenizer() -> None:
    raise RuntimeError("REAL NON-ASSERTION REGRESSION")


def _block_collecting_per_model_failures(*operations) -> None:
    """Run the `test_every_listed_model_counts_tokens` shape on the real `HubFailures`."""
    with skip_if_rate_limited() as hub:
        failures = HubFailures(hub)
        for operation in operations:
            with failures.collect(f"model-for-{operation.__name__}"):
                operation()
        failures.report("The following models failed to count tokens:")


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


def test_collected_failures_the_hub_caused_do_not_beat_the_skip(monkeypatch):
    """A loop that catches its own errors must not turn a 429 into a red build.

    Every entry here is the Hub refusing the fetch, so there is nothing to report and
    the skip is allowed through. Without attribution the `pytest.fail` built from those
    entries would be a verdict, and verdicts always win.
    """
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.skip.Exception, match="rate limit"):
        _block_collecting_per_model_failures(_refused_by_the_hub)


def test_collected_genuine_failure_beats_a_coincident_rate_limit(monkeypatch):
    """A real regression during a Hub outage still fails, naming only the real one."""
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _block_collecting_per_model_failures(_broken_tokenizer, _refused_by_the_hub)

    message = str(excinfo.value)
    assert "REAL NON-ASSERTION REGRESSION" in message
    assert "429 Client Error" not in message
    assert "plus 1 refused by the Hub" in message


def test_collected_genuine_failure_without_a_rate_limit_is_reported(monkeypatch):
    _stub_transport(monkeypatch, [200])
    with pytest.raises(pytest.fail.Exception, match="REAL NON-ASSERTION REGRESSION"):
        _block_collecting_per_model_failures(_broken_tokenizer)


def test_collected_failure_is_reported_when_the_hub_relented(monkeypatch):
    """A 429 the Hub went on to answer excuses nothing — no skip is coming."""
    _stub_transport(monkeypatch, [429, 200])
    with pytest.raises(pytest.fail.Exception, match="429 Client Error"):
        _block_collecting_per_model_failures(_refused_then_answered)


def test_skip_reason_names_an_exception_it_swallowed(monkeypatch):
    """Non-strict runs still skip, but the masked error is in the reason, not lost."""
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _block_failing_for_another_reason()

    assert "rate limit" in str(excinfo.value)
    assert "ValueError: server error" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_strict_mode_never_loses_a_swallowed_exception(monkeypatch):
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    _stub_transport(monkeypatch, [429])
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _block_failing_for_another_reason()

    assert "rate limit" in str(excinfo.value)
    assert "ValueError: server error" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
