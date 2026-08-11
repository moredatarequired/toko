"""Tests for API-based token counting (Anthropic, Google)."""

import json
import os

import httpx
import pytest
import respx

from toko.counter import ANTHROPIC_COUNT_URL, GOOGLE_COUNT_URL_BASE, count_tokens
from toko.result import TokenCount


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_count_tokens():
    """Test counting tokens with Anthropic API."""
    result = count_tokens("hello world", model="claude-haiku-4-5")
    assert result.count > 0
    # Anthropic should give similar count to OpenAI for simple text
    assert 1 <= result.count <= 10


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_different_models():
    """Test that different Anthropic models can count."""
    text = "The quick brown fox jumps over the lazy dog"
    sonnet_count = count_tokens(text, model="claude-sonnet-4-5").count
    haiku_count = count_tokens(text, model="claude-haiku-4-5").count

    assert sonnet_count > 0
    assert haiku_count > 0
    # Anthropic models should give similar counts
    assert abs(sonnet_count - haiku_count) <= 5


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_google_count_tokens():
    """Test counting tokens with Google API."""
    result = count_tokens("hello world", model="gemini-2.5-flash")
    assert result.count > 0
    assert 1 <= result.count <= 10


def test_missing_anthropic_key():
    """Test that missing Anthropic API key raises error."""
    original_key = os.environ.get("ANTHROPIC_API_KEY")
    if original_key:
        del os.environ["ANTHROPIC_API_KEY"]

    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            count_tokens("hello", model="claude-sonnet-4-5")
    finally:
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key


def test_missing_google_key():
    """Test that missing Google API key raises error."""
    original_key = os.environ.get("GOOGLE_API_KEY")
    if original_key:
        del os.environ["GOOGLE_API_KEY"]

    try:
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            count_tokens("hello", model="gemini-2.5-flash")
    finally:
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key


@respx.mock
def test_anthropic_unexpected_payload_raises_value_error(monkeypatch):
    """A malformed success response must raise the error type callers catch."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    respx.post(ANTHROPIC_COUNT_URL).mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    with pytest.raises(ValueError, match="Unexpected response from Anthropic"):
        count_tokens("hello", model="claude-sonnet-4-5", use_cache=False)


@respx.mock
def test_google_unexpected_payload_raises_value_error(monkeypatch):
    """A malformed success response must raise the error type callers catch."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    respx.post(url__startswith=GOOGLE_COUNT_URL_BASE).mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    with pytest.raises(ValueError, match="Unexpected response from Google"):
        count_tokens("hello", model="gemini-2.5-flash", use_cache=False)


@respx.mock
def test_anthropic_count_reports_the_canonical_model_the_alias_resolved_to(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = respx.post(ANTHROPIC_COUNT_URL).mock(
        return_value=httpx.Response(200, json={"input_tokens": 3})
    )

    counted = count_tokens("hello", model="claude-sonnet-4-5", use_cache=False)

    assert counted == TokenCount(
        count=3, model="claude-sonnet-4-5-20250929", provider="anthropic"
    )
    assert json.loads(route.calls.last.request.content)["model"] == counted.model
