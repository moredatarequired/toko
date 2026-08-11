"""Tests for optional provider support (transformers, mistral)."""

import os

import pytest

from tests.hf_hub import skip_if_rate_limited
from toko.counter import count_tokens


def check_token_counting(
    model_name: str, use_cache: bool = False, text: str = "Hello world, this is a test!"
) -> None:
    with skip_if_rate_limited():
        counted = count_tokens(text, model_name, use_cache=use_cache)
    assert counted.count > 0
    assert 5 <= counted.count <= 20


@pytest.mark.slow
def test_mistral_tokenization():
    """Test that Mistral tokenization works with mistral-common."""
    check_token_counting("mistral-small-latest", use_cache=False)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
def test_llama_tokenization():
    """Test that Llama tokenization works with HuggingFace transformers."""
    check_token_counting("tinyllama/TinyLlama-1.1B-Chat-v1.0", use_cache=False)


@pytest.mark.slow
def test_deepseek_tokenization():
    """Test that DeepSeek tokenization works with HuggingFace transformers."""
    # Note: This will download ~7-9MB tokenizer on first run
    check_token_counting("deepseek-ai/DeepSeek-V3", use_cache=False)


@pytest.mark.slow
def test_qwen_tokenization():
    """Test that Qwen tokenization works with HuggingFace transformers."""
    # Note: This will download ~7MB tokenizer on first run
    check_token_counting("Qwen/Qwen2.5-7B", use_cache=False)
