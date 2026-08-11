"""API keys must never reach a user-visible error message."""

import httpx
import pytest
import respx

from toko.counter import (
    ANTHROPIC_COUNT_URL,
    GOOGLE_COUNT_URL_BASE,
    _redact_key,
    count_tokens,
)

SENTINEL = "toko-test-sentinel-do-not-log"


@respx.mock
@pytest.mark.parametrize("status", [401, 404])
def test_google_http_error_does_not_leak_key(monkeypatch, status):
    monkeypatch.setenv("GOOGLE_API_KEY", SENTINEL)
    respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(status, json={"error": "nope"})
    )

    with pytest.raises(
        ValueError, match="Failed to count tokens for Google"
    ) as excinfo:
        count_tokens("hello", model="gemini-2.5-flash", use_cache=False)

    assert SENTINEL not in str(excinfo.value)


@respx.mock
def test_google_key_travels_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", SENTINEL)
    route = respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(200, json={"totalTokens": 3})
    )

    assert count_tokens("hello", model="gemini-2.5-flash", use_cache=False) == 3

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == SENTINEL
    assert SENTINEL not in str(request.url)


@respx.mock
def test_anthropic_key_with_trailing_newline_does_not_leak(monkeypatch):
    """`export ANTHROPIC_API_KEY=$(cat keyfile)` used to make httpx echo the key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"{SENTINEL}\n")
    route = respx.post(ANTHROPIC_COUNT_URL).mock(
        return_value=httpx.Response(401, json={"error": "nope"})
    )

    with pytest.raises(
        ValueError, match="Failed to count tokens for Anthropic"
    ) as excinfo:
        count_tokens("hello", model="claude-sonnet-4-5", use_cache=False)

    message = str(excinfo.value)
    assert SENTINEL not in message
    assert "Illegal header value" not in message
    assert route.calls.last.request.headers["x-api-key"] == SENTINEL


@respx.mock
def test_xai_api_error_does_not_leak_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", SENTINEL)
    monkeypatch.setattr("toko.counter.HAS_TRANSFORMERS", False)
    respx.post("https://api.x.ai/v1/tokenize").mock(
        return_value=httpx.Response(401, json={"error": "nope"})
    )

    with pytest.raises(ValueError, match="Failed to count tokens for xAI") as excinfo:
        count_tokens("hello", model="grok-4.5", use_cache=False)

    assert SENTINEL not in str(excinfo.value)


@pytest.mark.parametrize("empty", ["", None])
def test_redact_key_is_a_noop_for_empty_keys(empty):
    message = "Client error for url 'https://example.test/v1?key='"
    assert _redact_key(message, empty) == message


def test_redact_key_replaces_raw_and_encoded_forms():
    key = "abc/def+ghi"
    message = f"url 'https://example.test?key={key}' header {key}"
    redacted = _redact_key(message, key)

    assert key not in redacted
    assert redacted.count("***") == 2


def test_redact_key_replaces_stripped_key_inside_header_repr():
    """Httpx reports the raw bytes, where the newline shows up escaped."""
    key = f"{SENTINEL}\n"
    message = f"Illegal header value b'{SENTINEL}\\n'"

    assert SENTINEL not in _redact_key(message, key)


def test_redact_key_replaces_percent_encoded_key():
    key = "abc def"
    assert "abc%20def" not in _redact_key("url ?key=abc%20def", key)
