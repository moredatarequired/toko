"""Token counting logic."""

import tiktoken

from toko.models import get_model


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model: Model name

    Returns:
        Number of tokens

    Raises:
        ValueError: If model is not supported
    """
    model_info = get_model(model)

    if model_info.provider == "openai" and model_info.encoding:
        encoding = tiktoken.get_encoding(model_info.encoding)
        return len(encoding.encode(text))

    raise ValueError(f"Token counting not implemented for {model_info.provider}")
