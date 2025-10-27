"""Tests for API-based token counting (Anthropic, Google)."""

import os

import pytest

from toko.counter import count_tokens


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_count_tokens():
    """Test counting tokens with Anthropic API."""
    result = count_tokens("hello world", model="claude-haiku-4-5")
    assert result > 0
    # Anthropic should give similar count to OpenAI for simple text
    assert 1 <= result <= 10


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_anthropic_different_models():
    """Test that different Anthropic models can count."""
    text = "The quick brown fox jumps over the lazy dog"
    sonnet_count = count_tokens(text, model="claude-sonnet-4-5")
    haiku_count = count_tokens(text, model="claude-haiku-4-5")

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
    assert result > 0
    assert 1 <= result <= 10


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
