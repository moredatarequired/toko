"""Tests for token counting."""

import pytest

from toko.counter import count_tokens


def test_count_tokens_simple_text():
    """Test counting tokens in simple text."""
    result = count_tokens("hello world", model="gpt-5")
    assert result > 0
    assert result == 2


def test_count_tokens_empty_string():
    """Test counting tokens in empty string."""
    result = count_tokens("", model="gpt-5")
    assert result == 0


def test_count_tokens_different_models():
    """Test that different models can count the same text."""
    text = "The quick brown fox jumps over the lazy dog"
    gpt5_count = count_tokens(text, model="gpt-5")
    gpt5_mini_count = count_tokens(text, model="gpt-5-mini")

    # Both should return positive counts
    assert gpt5_count > 0
    assert gpt5_mini_count > 0
    # gpt-5 and gpt-5-mini share the same tokenizer so counts should match
    assert gpt5_count == gpt5_mini_count


def test_count_tokens_unicode():
    """Test counting tokens with unicode characters."""
    result = count_tokens("Hello 世界", model="gpt-5")
    assert result > 0


def test_count_tokens_unknown_model():
    """Test that unknown model raises error."""
    with pytest.raises(ValueError, match="Could not detect provider"):
        count_tokens("hello", model="unknown-model-xyz")


def test_count_tokens_gpt5_variants():
    """Test that gpt-5.x variants use the gpt-5 tokenizer."""
    text = "The quick brown fox jumps over the lazy dog"
    gpt5_count = count_tokens(text, model="gpt-5")
    gpt51_count = count_tokens(text, model="gpt-5.1")
    gpt52_count = count_tokens(text, model="gpt-5.2")

    # All variants should return the same count (same tokenizer)
    assert gpt5_count > 0
    assert gpt51_count == gpt5_count
    assert gpt52_count == gpt5_count
