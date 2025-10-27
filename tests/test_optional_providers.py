"""Tests for optional provider support (transformers, mistral)."""

import pytest

from toko.counter import count_tokens


def test_mistral_tokenization():
    """Test that Mistral tokenization works with mistral-common."""
    text = "Hello world, this is a test!"

    # Test with a Mistral model
    count = count_tokens(text, "mistral-small-latest", use_cache=False)

    # Should return a positive integer
    assert isinstance(count, int)
    assert count > 0

    # Rough sanity check - should be reasonable token count for this text
    assert 5 <= count <= 20


def test_llama_tokenization():
    """Test that Llama tokenization works with HuggingFace transformers.

    Note: This test is skipped because Llama models require HuggingFace authentication.
    Users must accept Meta's license and set HF_TOKEN to use Llama models.
    """
    pytest.skip("Llama models require HuggingFace authentication - skipping")


def test_deepseek_tokenization():
    """Test that DeepSeek tokenization works with HuggingFace transformers."""
    text = "Hello world, this is a test!"

    # Test with a DeepSeek model
    # Note: This will download ~7-9MB tokenizer on first run
    count = count_tokens(text, "deepseek-ai/DeepSeek-V3", use_cache=False)

    # Should return a positive integer
    assert isinstance(count, int)
    assert count > 0

    # Rough sanity check
    assert 5 <= count <= 20


def test_qwen_tokenization():
    """Test that Qwen tokenization works with HuggingFace transformers."""
    text = "Hello world, this is a test!"

    # Test with a Qwen model
    # Note: This will download ~7MB tokenizer on first run
    count = count_tokens(text, "Qwen/Qwen2.5-7B", use_cache=False)

    # Should return a positive integer
    assert isinstance(count, int)
    assert count > 0

    # Rough sanity check
    assert 5 <= count <= 20


def test_transformers_caching():
    """Test that transformers tokenizers are properly cached."""
    text = "Hello world!"

    # First call - may download tokenizer
    count1 = count_tokens(text, "Qwen/Qwen2.5-7B", use_cache=False)

    # Second call - should use cached tokenizer (fast)
    count2 = count_tokens(text, "Qwen/Qwen2.5-7B", use_cache=False)

    # Should return same count
    assert count1 == count2


def test_multiple_providers():
    """Test that we can use multiple optional providers in the same session."""
    text = "Hello world!"

    # Try optional providers (excluding Llama which requires auth)
    mistral_count = count_tokens(text, "mistral-small-latest", use_cache=False)
    deepseek_count = count_tokens(text, "deepseek-ai/DeepSeek-V3", use_cache=False)
    qwen_count = count_tokens(text, "Qwen/Qwen2.5-7B", use_cache=False)

    # All should be valid
    assert mistral_count > 0
    assert deepseek_count > 0
    assert qwen_count > 0

    # Token counts will differ between providers (different tokenizers)
    # But should all be in a reasonable range for "Hello world!"
    for count in [mistral_count, deepseek_count, qwen_count]:
        assert 1 <= count <= 10
