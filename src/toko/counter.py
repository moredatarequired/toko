"""Token counting logic."""

import os

import google.generativeai as genai
import tiktoken
from anthropic import Anthropic

from toko.models import get_model


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model: Model name

    Returns:
        Number of tokens

    Raises:
        ValueError: If model is not supported or API key is missing
    """
    model_info = get_model(model)

    # OpenAI and xAI use tiktoken
    if model_info.provider in ("openai", "xai") and model_info.encoding:
        encoding = tiktoken.get_encoding(model_info.encoding)
        return len(encoding.encode(text))

    if model_info.provider == "anthropic":
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
        return result.input_tokens

    if model_info.provider == "google":
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
        return result.total_tokens

    raise ValueError(f"Token counting not implemented for {model_info.provider}")
