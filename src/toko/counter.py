"""Token counting logic."""

import os
from typing import TYPE_CHECKING, Protocol, cast

# Suppress transformers warning about missing PyTorch/TF/Flax
# We only need tokenizers, not the full ML frameworks
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import google.generativeai as genai
import tiktoken
from anthropic import Anthropic

from toko.cache import cache_count, get_cached_count
from toko.models import ModelInfo, get_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers import PreTrainedTokenizerBase


# Check for optional dependencies without importing them
# (importing transformers triggers a warning if PyTorch/TF/Flax not installed)
try:
    import importlib.util

    HAS_MISTRAL = importlib.util.find_spec("mistral_common") is not None
except ImportError:
    HAS_MISTRAL = False

try:
    import importlib.util

    HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
except ImportError:
    HAS_TRANSFORMERS = False

# Cache tokenizers at module level to avoid reloading on every call
_TOKENIZER_CACHE: dict[str, object] = {}


class TokenizerProtocol(Protocol):
    """Minimal interface expected from tokenizer implementations."""

    def encode(self, text: str, /, *args: object, **kwargs: object) -> list[int]:
        """Encode text into token identifiers."""
        ...


def _get_tiktoken_encoding_for_model(model_name: str) -> TokenizerProtocol | None:
    """Return a tiktoken encoding for a specific model name, if available."""
    cache_key = f"tiktoken:model:{model_name}"
    cached = _TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cast("TokenizerProtocol", cached)

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except (KeyError, ValueError):
        return None

    tokenizer = cast("TokenizerProtocol", encoding)
    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _get_tiktoken_encoding_by_name(
    encoding_name: str,
) -> TokenizerProtocol | None:
    """Return a tiktoken encoding by canonical encoding name."""
    cache_key = f"tiktoken:encoding:{encoding_name}"
    cached = _TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        return cast("TokenizerProtocol", cached)

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        return None

    tokenizer = cast("TokenizerProtocol", encoding)
    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def _count_with_tiktoken(text: str, model_name: str) -> int | None:
    """Try to count tokens using tiktoken for the given model name."""
    encoding = _get_tiktoken_encoding_for_model(model_name)
    if encoding is None:
        return None
    return len(encoding.encode(text))


def _count_with_provider(text: str, model_info: ModelInfo) -> int:
    handler_obj = _PROVIDER_HANDLERS.get(model_info.provider)
    if handler_obj is None:
        raise ValueError(
            f"Token counting not supported for provider: {model_info.provider}. "
            "Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Llama, DeepSeek, Qwen"
        )
    handler = cast("Callable[[str, ModelInfo], int]", handler_obj)
    return handler(text, model_info)


def _count_openai(_text: str, model_info: ModelInfo) -> int:
    raise ValueError(
        f"tiktoken does not recognize model '{model_info.name}'. "
        "Install the latest tiktoken or verify the model name."
    )


def _count_xai(text: str, model_info: ModelInfo) -> int:
    encoding_name = model_info.encoding
    encoding = (
        _get_tiktoken_encoding_for_model(model_info.name)
        if encoding_name is None
        else _get_tiktoken_encoding_by_name(encoding_name)
    )
    if encoding is None:
        raise ValueError(
            f"tiktoken does not include an encoding for xAI model '{model_info.name}'. "
            "Install the latest tiktoken or verify the model name."
        )
    return len(encoding.encode(text))


def _count_anthropic(text: str, model_info: ModelInfo) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it or add to ~/.config/toko/config.toml"
        )
    client = Anthropic(api_key=api_key)
    try:
        result = client.messages.count_tokens(
            model=model_info.name,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as e:
        raise ValueError(
            f"Failed to count tokens for Anthropic model {model_info.name}: {e}. "
            "The model may not exist or may not be available with your API key."
        ) from e

    return result.input_tokens


def _count_google(text: str, model_info: ModelInfo) -> int:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Set it or add to ~/.config/toko/config.toml"
        )
    genai.configure(api_key=api_key)
    try:
        google_model = genai.GenerativeModel(model_info.name)
        result = google_model.count_tokens(text)
    except Exception as e:
        raise ValueError(
            f"Failed to count tokens for Google model {model_info.name}: {e}. "
            "The model may not exist or may not support token counting."
        ) from e

    return result.total_tokens


def _count_mistral(text: str, model_info: ModelInfo) -> int:
    if not HAS_MISTRAL:
        raise ValueError(
            "Mistral models require the 'mistral-common' package. "
            "Install with: uv tool install 'toko[mistral]' or uv add 'toko[mistral]'"
        )

    try:
        from mistral_common.protocol.instruct.messages import (  # noqa: PLC0415
            UserMessage,
        )
        from mistral_common.protocol.instruct.request import (  # noqa: PLC0415
            ChatCompletionRequest,
        )
        from mistral_common.tokens.tokenizers.mistral import (  # noqa: PLC0415
            MistralTokenizer,
        )
    except Exception as e:
        raise ValueError(f"Failed to import mistral-common: {e}") from e

    cache_key = f"mistral:{model_info.name}"
    if cache_key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[cache_key] = MistralTokenizer.from_model(model_info.name)
    tokenizer = cast("MistralTokenizer", _TOKENIZER_CACHE[cache_key])

    request = ChatCompletionRequest(messages=[UserMessage(content=text)])
    tokens = tokenizer.encode_chat_completion(request).tokens
    return len(tokens)


def _count_transformers(text: str, model_info: ModelInfo) -> int:
    if not HAS_TRANSFORMERS:
        raise ValueError(
            f"{model_info.provider.capitalize()} models require the 'transformers' package. "
            "Install with: uv tool install 'toko[transformers]' or uv add 'toko[transformers]'"
        )

    cache_key = f"transformers:{model_info.name}"
    if cache_key not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer  # noqa: PLC0415

        _TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
            model_info.name,
            trust_remote_code=True,
        )

    tokenizer = cast("PreTrainedTokenizerBase", _TOKENIZER_CACHE[cache_key])
    try:
        tokens = tokenizer.encode(text)
    except Exception as e:
        error_str = str(e)
        if "is not a local folder and is not a valid model identifier" in error_str:
            examples = {
                "qwen": "Qwen/Qwen3-8B, Qwen/Qwen2.5-7B",
                "deepseek": "deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1",
                "llama": "meta-llama/Llama-3.2-1B, meta-llama/Meta-Llama-3-8B",
            }
            example_hint = examples.get(model_info.provider)
            hint = f" Try: {example_hint}" if example_hint else ""
            raise ValueError(
                f"Model '{model_info.name}' not found on HuggingFace. "
                f"Use the full model path (org/model-name).{hint}"
            ) from e
        if "401" in error_str or "authentication" in error_str.lower():
            raise ValueError(
                f"Model '{model_info.name}' requires authentication. "
                "Set HF_TOKEN environment variable or run: huggingface-cli login"
            ) from e
        raise ValueError(
            f"Failed to count tokens for {model_info.provider.capitalize()} model {model_info.name}: {error_str}"
        ) from e

    return len(tokens)


_PROVIDER_HANDLERS: dict[str, object] = {
    "openai": _count_openai,
    "xai": _count_xai,
    "anthropic": _count_anthropic,
    "google": _count_google,
    "mistral": _count_mistral,
}

for provider in ("llama", "deepseek", "qwen"):
    _PROVIDER_HANDLERS[provider] = _count_transformers


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

    token_count = _count_with_tiktoken(text, model)
    model_info = None

    if token_count is None:
        model_info = get_model(model)
        if model_info.name != model:
            token_count = _count_with_tiktoken(text, model_info.name)

    if token_count is not None:
        if use_cache:
            cache_count(text, model, token_count)
            if model_info and model_info.name != model:
                cache_count(text, model_info.name, token_count)
        return token_count

    if model_info is None:
        model_info = get_model(model)

    token_count = _count_with_provider(text, model_info)

    if use_cache:
        cache_count(text, model, token_count)
        if model_info.name != model:
            cache_count(text, model_info.name, token_count)

    return token_count
