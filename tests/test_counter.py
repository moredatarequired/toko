"""Tests for token counting."""

import pytest

from toko.counter import count_tokens


def test_count_tokens_simple_text():
    """Test counting tokens in simple text."""
    result = count_tokens("hello world", model="gpt-4o")
    assert result > 0
    assert result == 2


def test_count_tokens_empty_string():
    """Test counting tokens in empty string."""
    result = count_tokens("", model="gpt-4o")
    assert result == 0


def test_count_tokens_different_models():
    """Test that different models can count the same text."""
    text = "The quick brown fox jumps over the lazy dog"
    gpt4o_count = count_tokens(text, model="gpt-4o")
    gpt4o_mini_count = count_tokens(text, model="gpt-4o-mini")

    # Both should return positive counts
    assert gpt4o_count > 0
    assert gpt4o_mini_count > 0
    # For GPT-4 models, they use the same tokenizer so counts should be equal
    assert gpt4o_count == gpt4o_mini_count


def test_count_tokens_unicode():
    """Test counting tokens with unicode characters."""
    result = count_tokens("Hello 世界", model="gpt-4o")
    assert result > 0


def test_count_tokens_unknown_model():
    """Test that unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        count_tokens("hello", model="unknown-model-xyz")
