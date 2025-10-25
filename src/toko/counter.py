"""Token counting logic."""

import os

import google.generativeai as genai
import tiktoken
from anthropic import Anthropic

from toko.cache import cache_count, get_cached_count
from toko.models import get_model


def count_tokens(text: str, model: str, *, use_cache: bool = True) -> int:
    """Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model: Model name
        use_cache: Whether to use caching (default True)

    Returns:
        Number of tokens

    Raises:
        ValueError: If model is not supported or API key is missing
    """
    # Check cache first
    if use_cache:
        cached = get_cached_count(text, model)
        if cached is not None:
            return cached

    model_info = get_model(model)

    # Count tokens based on provider
    token_count: int

    # OpenAI and xAI use tiktoken
    if model_info.provider in ("openai", "xai") and model_info.encoding:
        encoding = tiktoken.get_encoding(model_info.encoding)
        token_count = len(encoding.encode(text))
    elif model_info.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it or add to ~/.config/toko/config.toml"
            )
        client = Anthropic(api_key=api_key)
        result = client.messages.count_tokens(
            model=model_info.name,
            messages=[{"role": "user", "content": text}],
        )
        token_count = result.input_tokens
    elif model_info.provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Set it or add to ~/.config/toko/config.toml"
            )
        genai.configure(api_key=api_key)
        # Google uses a different API - count_tokens on the model
        google_model = genai.GenerativeModel(model_info.name)
        result = google_model.count_tokens(text)
        token_count = result.total_tokens
    else:
        raise ValueError(f"Token counting not implemented for {model_info.provider}")

    # Cache the result
    if use_cache:
        cache_count(text, model, token_count)

    return token_count
