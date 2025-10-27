"""Token counting logic."""

import importlib
import importlib.util
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

# Suppress transformers warning about missing PyTorch/TF/Flax
# We only need tokenizers, not the full ML frameworks
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import httpx
import tiktoken

from toko.cache import cache_count, get_cached_count
from toko.models import ModelInfo, get_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers import PreTrainedTokenizerBase


# Check for optional dependencies without importing them
# (importing transformers triggers a warning if PyTorch/TF/Flax not installed)
try:
    HAS_MISTRAL = importlib.util.find_spec("mistral_common") is not None
except ImportError:
    HAS_MISTRAL = False

try:
    HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
except ImportError:
    HAS_TRANSFORMERS = False


@lru_cache(maxsize=1)
def _configure_transformers_logging() -> None:
    if not HAS_TRANSFORMERS:
        return
    try:
        hf_logging = importlib.import_module("transformers.utils.logging")
    except Exception:
        return
    if hasattr(hf_logging, "set_verbosity_error"):
        hf_logging.set_verbosity_error()


# Cache tokenizers at module level to avoid reloading on every call
_TOKENIZER_CACHE: dict[str, object] = {}

ANTHROPIC_COUNT_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_API_VERSION = "2023-06-01"
GOOGLE_COUNT_URL_BASE = "https://generativelanguage.googleapis.com/v1beta"


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


def _get_tiktoken_encoding_by_name(encoding_name: str) -> TokenizerProtocol | None:
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
    api_key = os.environ.get("XAI_API_KEY")
    if api_key:
        try:
            return _count_xai_via_api(text, model_info.name, api_key)
        except Exception as api_error:
            last_error = api_error
        else:
            last_error = None
    else:
        last_error = ValueError(
            "XAI_API_KEY environment variable not set. Falling back to Hugging Face tokenizer."
        )

    try:
        return _count_xai_via_transformers(text)
    except Exception as hf_error:
        message = (
            f"Failed to count tokens for xAI model {model_info.name}. "
            "Provide XAI_API_KEY for API-based counting, or install 'toko[transformers]' "
            "and ensure HF_TOKEN grants access to Xenova/grok-1-tokenizer."
        )
        if api_key and last_error is not None:
            raise ValueError(f"{message} Last API error: {last_error}") from hf_error
        raise ValueError(message) from hf_error


def _count_xai_via_api(text: str, model_name: str, api_key: str) -> int:
    try:
        response = httpx.post(
            "https://api.x.ai/v1/tokenize",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model_name, "input": text},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ValueError(f"xAI tokenization request failed: {exc}") from exc

    count = _extract_token_count(data)
    if count is None:
        raise ValueError(f"Unexpected response from xAI token API: {data!r}")
    return count


def _extract_token_count(payload: object) -> int | None:
    if isinstance(payload, dict):
        data = cast("dict[str, object]", payload)
        for key in ("token_count", "count"):
            value = data.get(key)
            if isinstance(value, int):
                return value
        usage = data.get("usage")
        if isinstance(usage, dict):
            usage_dict = cast("dict[str, object]", usage)
            for key in ("input_tokens", "prompt_tokens", "total_tokens"):
                value = usage_dict.get(key)
                if isinstance(value, int):
                    return value
        data_field = data.get("data")
        if isinstance(data_field, dict):
            return _extract_token_count(data_field)
        if isinstance(data_field, list):
            for item in data_field:
                result = _extract_token_count(item)
                if result is not None:
                    return result
    return None


def _count_xai_via_transformers(text: str) -> int:
    if not HAS_TRANSFORMERS:
        raise ValueError(
            "transformers package not available. Install with: uv tool install 'toko[transformers]'"
        )

    _configure_transformers_logging()

    cache_key = "transformers:xai:grok-1"
    if cache_key not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer  # noqa: PLC0415

        _TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
            "Xenova/grok-1-tokenizer", trust_remote_code=True
        )

    tokenizer = cast("PreTrainedTokenizerBase", _TOKENIZER_CACHE[cache_key])
    tokens = tokenizer.encode(text)
    return len(tokens)


def _count_anthropic(text: str, model_info: ModelInfo) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it or add to ~/.config/toko/config.toml"
        )
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_API_VERSION,
    }
    payload = {
        "model": model_info.name,
        "messages": [{"role": "user", "content": text}],
    }
    try:
        response = httpx.post(
            ANTHROPIC_COUNT_URL, headers=headers, json=payload, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(
            f"Failed to count tokens for Anthropic model {model_info.name}: {exc}. "
            "The model may not exist or may not be available with your API key."
        ) from exc

    input_tokens = data.get("input_tokens")
    if not isinstance(input_tokens, int):
        raise TypeError(f"Unexpected response from Anthropic token API: {data!r}")
    return input_tokens


def _count_google(text: str, model_info: ModelInfo) -> int:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Set it or add to ~/.config/toko/config.toml"
        )
    url = f"{GOOGLE_COUNT_URL_BASE}/{model_info.name}:countTokens"
    payload = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=10.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(
            f"Failed to count tokens for Google model {model_info.name}: {exc}. "
            "The model may not exist or may not support token counting."
        ) from exc

    total_tokens = data.get("totalTokens")
    if not isinstance(total_tokens, int):
        raise TypeError(f"Unexpected response from Google token API: {data!r}")
    return total_tokens


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

    _configure_transformers_logging()

    try:
        cache_key = f"transformers:{model_info.name}"
        if cache_key not in _TOKENIZER_CACHE:
            from transformers import AutoTokenizer  # noqa: PLC0415

            _TOKENIZER_CACHE[cache_key] = AutoTokenizer.from_pretrained(
                model_info.name, trust_remote_code=True
            )

        tokenizer = cast("PreTrainedTokenizerBase", _TOKENIZER_CACHE[cache_key])
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
        if "gated" in error_str.lower() or "access to model" in error_str.lower():
            raise ValueError(
                f"Model '{model_info.name}' is gated on Hugging Face. Accept the license"
                " and provide HF_TOKEN or run: huggingface-cli login"
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

_PROVIDER_HANDLERS["huggingface"] = _count_transformers


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

    token_count: int | None = None
    if model_info.provider == "openai":
        names_to_try = []
        if model_info.name != model:
            names_to_try.append(model_info.name)
        names_to_try.append(model)
        for name in names_to_try:
            token_count = _count_with_tiktoken(text, name)
            if token_count is not None:
                break

    if token_count is None:
        token_count = _count_with_provider(text, model_info)

    if use_cache:
        cache_count(text, model, token_count)
        if model_info.name != model:
            cache_count(text, model_info.name, token_count)

    return token_count
