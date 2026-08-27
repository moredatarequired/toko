"""Token counting logic."""

import contextlib
import importlib
import importlib.util
import os
import sys
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import quote

# Suppress transformers warning about missing PyTorch/TF/Flax
# We only need tokenizers, not the full ML frameworks
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import httpx
import tiktoken

from toko.cache import cache_count, get_cached_count
from toko.models import ModelInfo, get_model, retirement_notice
from toko.result import TokenCount

if TYPE_CHECKING:
    from collections.abc import Callable

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


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

# Counts run concurrently, so the miss below has to be held under a lock: threads that
# all miss the same key would otherwise each build and keep their own tokenizer, and a
# tokenizer is hundreds of megabytes. One lock per key rather than one for the cache,
# so a slow load -- a first-time HuggingFace download -- only blocks the threads that
# want that same tokenizer.
_TOKENIZER_LOCKS: dict[str, threading.Lock] = {}
_TOKENIZER_LOCKS_LOCK = threading.Lock()


def _cached_tokenizer[T](cache_key: str, load: Callable[[], T]) -> T:
    cached = _TOKENIZER_CACHE.get(cache_key)
    if cached is None:
        with _TOKENIZER_LOCKS_LOCK:
            lock = _TOKENIZER_LOCKS.setdefault(cache_key, threading.Lock())
        with lock:
            cached = _TOKENIZER_CACHE.get(cache_key)
            if cached is None:
                cached = _TOKENIZER_CACHE[cache_key] = load()
    return cast("T", cached)


# (warning kind, model name) pairs already emitted, so counting a directory does not
# repeat the same notice once per file. The kind is part of the key so that two
# different warnings about one model cannot suppress each other.
_WARNED_ONCE: set[tuple[str, str]] = set()

# Counts run concurrently, so the test and the insert below have to be one step: two
# threads that both miss the set would otherwise both print the same notice, and two
# printing at once can interleave halfway through a line.
_WARN_LOCK = threading.Lock()


def _warn_once(kind: str, model_name: str, message: str) -> None:
    key = (kind, model_name)
    with _WARN_LOCK:
        if key in _WARNED_ONCE:
            return
        _WARNED_ONCE.add(key)
        print(f"Warning: {message}", file=sys.stderr)


# Below this length a "key" is more likely to be a common substring than a secret,
# and blanking it out mangles the surrounding text without protecting anything.
_MIN_REDACTABLE_KEY_LENGTH = 8


def _redact_key(message: str, api_key: str | None) -> str:
    """Strip an API key out of text that is about to be shown to the user.

    Applied to anything derived from a provider response, since an error body can
    echo the credential it rejected. The key is replaced in its verbatim,
    percent-encoded and backslash-escaped forms -- the last because a key holding a
    control character reaches the message as the two characters of an escape sequence.

    Never raises, whatever the key contains: callers are already handling a failure,
    and an exception here would both discard their message and carry the key in its
    own payload.
    """
    if not api_key:
        return message
    needles = {api_key, api_key.strip()}
    for needle in tuple(needles):
        # A key holding a byte that is not valid UTF-8 reaches us as a surrogate,
        # because that is how os.environ decodes one, and quote() cannot encode a
        # surrogate as UTF-8. surrogateescape turns it back into the original byte;
        # a value it still refuses simply contributes no encoded form, since a
        # redaction helper that raises would destroy the message it was sanitising.
        with contextlib.suppress(UnicodeError):
            needles.add(quote(needle, safe="", errors="surrogateescape"))
    needles |= {repr(needle)[1:-1] for needle in tuple(needles)}
    for needle in sorted(needles, key=len, reverse=True):
        if len(needle) >= _MIN_REDACTABLE_KEY_LENGTH:
            message = message.replace(needle, "***")
    return message


def _describe_request_failure(exc: Exception, url: str, api_key: str | None) -> str:
    """Say what a failed request did without quoting the transport's own message.

    httpx echoes back whatever it was handed -- the request URL for a key sent as a
    query parameter, the raw header bytes for a key it refused to send -- and no
    search-and-replace can reliably undo that, because a key holding one control
    character appears in the message as the two characters of an escape sequence.
    Reporting only the status code and the exception type removes the whole class,
    bar the one carve-out below where the transport's message is provably an errno.

    `api_key` has no default on purpose: a defaulted one lets a call site drop the
    argument and disable redaction without anything failing.
    """
    endpoint = url.split("?", 1)[0].split("#", 1)[0]
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        phrase = httpx.codes.get_reason_phrase(status) or "response"
        return f"HTTP {status} {phrase} from {endpoint}"
    if isinstance(exc, httpx.ConnectError):
        # The one exception to the rule above: a ConnectError is raised before any
        # request bytes exist, and its text is the OS-level cause behind the failed
        # socket or name lookup, so it cannot hold the key. Without it a DNS failure
        # and a refused connection read identically. Redacted regardless.
        cause = _redact_key(str(exc), api_key).strip()
        if cause:
            return f"ConnectError contacting {endpoint}: {cause}"
    return f"{type(exc).__name__} contacting {endpoint}"


def _api_key_from_env(env_var: str) -> str | None:
    # `export KEY=$(cat keyfile)` leaves a trailing newline, which httpx rejects
    # outright as a header value rather than sending.
    value = os.environ.get(env_var)
    return value.strip() if value else value


OPENAI_FALLBACK_ENCODING = "o200k_base"

# The tokenizer each named Mistral release shipped with, as a
# "<mistral-common classmethod>[+flag...]" spec for _load_mistral_tokenizer. A name has
# an entry here only if it identifies one release, so the tokenizer cannot drift out from
# under it; rolling names fall through to MISTRAL_FALLBACK_TOKENIZER instead. The dated
# entries mirror mistral-common's own MODEL_NAME_TO_TOKENIZER_CLS, which
# test_mistral_specs_match_mistral_common checks entry by entry -- toko keeps a copy
# because that table serves the deprecated `from_model` and shrinks with it, 1.10.0
# having already dropped every undated key below.
MISTRAL_TOKENIZERS = {
    "codestral-22b": "v3",
    "codestral-2405": "v3",
    "codestral-mamba-2407": "v3",
    "ministral-8b-2410": "v3+tekken",
    "mistral-embed": "v1",
    "mistral-large-2402": "v2",
    "mistral-large-2407": "v3",
    "mistral-large-2411": "v7",
    "mistral-large-v1": "v2",
    "mistral-medium-2312": "v1",
    "mistral-nemo": "v3+tekken",
    "mistral-small-2312": "v2",
    "mistral-small-2402": "v2",
    "mistral-small-2409": "v3+tekken",
    "mistral-small-v1": "v2",
    "mistral-tiny-2312": "v2",
    "mistral-tiny-2407": "v3",
    "open-mistral-7b": "v1",
    "open-mistral-nemo-2407": "v3+tekken",
    "open-mixtral-8x22b": "v3",
    "open-mixtral-8x22b-2404": "v3",
    "open-mixtral-8x7b": "v1",
    "pixtral": "v3+tekken+mm",
    "pixtral-12b-2409": "v3+tekken+mm",
    "pixtral-large": "v7+mm",
    "pixtral-large-2411": "v7+mm",
}

# mistral-common bundles nothing newer than November 2024, so no rolling alias
# ("mistral-large-latest") and no recent release can be counted with the tokenizer it
# actually ships. Every Mistral model since Nemo tokenizes with tekken, so tekken is a
# far better answer for an unrecognised name than refusing to count it -- flagged
# approximate, as the vocabulary is right but the tokenizer revision is not.
MISTRAL_FALLBACK_TOKENIZER = "v3+tekken"

ANTHROPIC_COUNT_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_API_VERSION = "2023-06-01"
GOOGLE_COUNT_URL_BASE = "https://generativelanguage.googleapis.com/v1beta"
XAI_TOKENIZE_URL = "https://api.x.ai/v1/tokenize"


class TokenizerProtocol(Protocol):
    """Minimal interface expected from tokenizer implementations."""

    def encode(self, text: str, /, *args: object, **kwargs: object) -> list[int]:
        """Encode text into token identifiers."""
        ...


def _exact(count: int, model_info: ModelInfo) -> TokenCount:
    return TokenCount(count=count, model=model_info.name, provider=model_info.provider)


def _approximate(count: int, model_info: ModelInfo, caveat: str) -> TokenCount:
    return TokenCount(
        count=count,
        model=model_info.name,
        provider=model_info.provider,
        approximate=True,
        caveat=caveat,
    )


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


def _count_with_provider(text: str, model_info: ModelInfo) -> TokenCount:
    handler = _PROVIDER_HANDLERS.get(model_info.provider)
    if handler is None:
        raise ValueError(
            f"Token counting not supported for provider: {model_info.provider}. "
            "Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Llama, DeepSeek, Qwen"
        )
    return handler(text, model_info)


def _warn_openai_estimate(model_name: str, encoding_name: str) -> str:
    caveat = f"unknown OpenAI model '{model_name}'; estimating with {encoding_name}"
    _warn_once("openai-estimate", model_name, caveat)
    return caveat


def _count_openai(text: str, model_info: ModelInfo) -> TokenCount:
    # tiktoken's own table is lowercase, and so is the OPENAI_MODEL_ENCODINGS lookup
    # in _build_openai_model; without matching here, 'GPT-5' would be called unknown.
    encoding = _get_tiktoken_encoding_for_model(model_info.name.lower())
    if encoding is not None:
        return _exact(len(encoding.encode(text)), model_info)

    # Every OpenAI model since gpt-4o uses o200k_base, so an unreleased name is
    # far better served by that than by refusing to count it.
    encoding_name = model_info.encoding or OPENAI_FALLBACK_ENCODING
    encoding = _get_tiktoken_encoding_by_name(encoding_name)
    if encoding is None:
        raise ValueError(
            f"tiktoken could not load encoding '{encoding_name}' for model "
            f"'{model_info.name}'. Install the latest tiktoken or verify the model name."
        )
    if model_info.encoding is not None:
        return _exact(len(encoding.encode(text)), model_info)

    caveat = _warn_openai_estimate(model_info.name, encoding_name)
    return _approximate(len(encoding.encode(text)), model_info, caveat)


def preload_tokenizer(model: str) -> None:
    """Build the local tokenizer for one model now, so no later count is the first build.

    tiktoken builds its encoding registry on first use, and a build that happens while
    the process is out of file descriptors leaves that registry permanently empty --
    pkgutil swallows the OSError, so tiktoken's own reset never fires, and every later
    lookup raises. Doing it here keeps the one fragile build in the caller's thread
    before any worker pool exists, where nothing has had a chance to exhaust anything.

    Only the OpenAI provider has a tokenizer to warm; the API-backed providers have
    nothing local, and the HuggingFace and Mistral ones are already built once under a
    lock and are far too large to load speculatively.

    The whole body is suppressed, not just the encode: warming is an optimisation, and
    a run that would have succeeded without it must still succeed. Resolving an
    encoding can reach the network -- tiktoken downloads its BPE file when its blob
    cache is cold -- and a run whose every count is already in toko's own cache never
    needed that download at all. `Exception` rather than a bare `except`, so
    KeyboardInterrupt and SystemExit still stop the run.
    """
    with contextlib.suppress(Exception):
        model_info = get_model(model)
        if model_info.provider != "openai":
            return

        # Resolved through tiktoken's own name table rather than any richer toko lookup:
        # warming only has to name the right BPE file, so the less of the counting path
        # it borrows, the less there is to keep in step with it. The table is a plain
        # dict lookup, so unlike encoding_for_model it builds nothing to answer.
        try:
            encoding_name = tiktoken.encoding_name_for_model(model_info.name.lower())
        except KeyError:
            encoding_name = model_info.encoding or OPENAI_FALLBACK_ENCODING

        encoding = _get_tiktoken_encoding_by_name(encoding_name)
        if encoding is None:
            return

        # Constructing the encoding is not the whole of loading it: tiktoken defers
        # `import regex` to the first encode (tiktoken/core.py), and an import needs a
        # descriptor of its own. Without this the registry survives a shortage but the
        # first real encode still dies on that import, inside a pool worker.
        encoding.encode("")


def _warn_if_retired(model_info: ModelInfo) -> None:
    notice = retirement_notice(model_info)
    if notice is not None:
        _warn_once("retired", model_info.name, notice)


def _warn_approximate(model_name: str, reason: str) -> str:
    caveat = (
        f"{reason}, so {model_name} was counted with the Grok-1 Hugging Face "
        "tokenizer. This count is approximate, not exact."
    )
    _warn_once("xai-approximate", model_name, caveat)
    return caveat


def _count_xai(text: str, model_info: ModelInfo) -> TokenCount:
    api_key = _api_key_from_env("XAI_API_KEY")
    if api_key:
        try:
            return _exact(
                _count_xai_via_api(text, model_info.name, api_key), model_info
            )
        except Exception as api_error:
            last_error = api_error
    else:
        last_error = None

    try:
        count = _count_xai_via_transformers(text)
    except Exception as hf_error:
        message = (
            f"Failed to count tokens for xAI model {model_info.name}. "
            "Provide XAI_API_KEY for API-based counting, or install 'toko[transformers]' "
            "and ensure HF_TOKEN grants access to Xenova/grok-1-tokenizer."
        )
        if api_key and last_error is not None:
            detail = _redact_key(str(last_error), api_key)
            raise ValueError(f"{message} Last API error: {detail}") from hf_error
        raise ValueError(message) from hf_error

    reason = (
        f"the xAI token API was unavailable ({_redact_key(str(last_error), api_key)})"
        if last_error is not None
        else "XAI_API_KEY is not set"
    )
    caveat = _warn_approximate(model_info.name, reason)
    return _approximate(count, model_info, caveat)


def _count_xai_via_api(text: str, model_name: str, api_key: str) -> int:
    try:
        response = httpx.post(
            XAI_TOKENIZE_URL,
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
        raise ValueError(
            f"xAI tokenization request failed: "
            f"{_describe_request_failure(exc, XAI_TOKENIZE_URL, api_key)}"
        ) from exc

    count = _extract_token_count(data)
    if count is None:
        # The body can echo the key back; _count_xai redacts before displaying this.
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

    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = _cached_tokenizer(
        "transformers:xai:grok-1",
        lambda: AutoTokenizer.from_pretrained(
            "Xenova/grok-1-tokenizer", trust_remote_code=True
        ),
    )
    tokens = tokenizer.encode(text)
    return len(tokens)


def _count_anthropic(text: str, model_info: ModelInfo) -> TokenCount:
    api_key = _api_key_from_env("ANTHROPIC_API_KEY")
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
            f"Failed to count tokens for Anthropic model {model_info.name}: "
            f"{_describe_request_failure(exc, ANTHROPIC_COUNT_URL, api_key)}. "
            "The model may not exist or may not be available with your API key."
        ) from exc

    # A 200 whose body is a JSON list or string has no .get; the resulting
    # AttributeError is not a ValueError, so it escapes the CLI's handler and
    # renders as a traceback.
    input_tokens = data.get("input_tokens") if isinstance(data, dict) else None
    if not isinstance(input_tokens, int):
        # TRY004 wants TypeError, but this is a malformed API payload rather than a
        # caller passing the wrong type, and callers (the CLI included) handle
        # ValueError — a TypeError here escapes as a traceback.
        raise ValueError(  # noqa: TRY004
            f"Unexpected response from Anthropic token API for {model_info.name}: "
            f"no integer 'input_tokens' field in {_redact_key(repr(data), api_key)}"
        )
    return _exact(input_tokens, model_info)


def _count_google(text: str, model_info: ModelInfo) -> TokenCount:
    api_key = _api_key_from_env("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Set it or add to ~/.config/toko/config.toml"
        )
    url = f"{GOOGLE_COUNT_URL_BASE}/{model_info.name}:countTokens"
    payload = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
    try:
        response = httpx.post(
            url, headers={"x-goog-api-key": api_key}, json=payload, timeout=10.0
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(
            f"Failed to count tokens for Google model {model_info.name}: "
            f"{_describe_request_failure(exc, url, api_key)}. "
            "The model may not exist or may not support token counting."
        ) from exc

    total_tokens = data.get("totalTokens") if isinstance(data, dict) else None
    if not isinstance(total_tokens, int):
        # See _count_anthropic: ValueError is the type callers are written to catch,
        # and a non-dict body would otherwise raise AttributeError above.
        raise ValueError(  # noqa: TRY004
            f"Unexpected response from Google token API for {model_info.name}: "
            f"no integer 'totalTokens' field in {_redact_key(repr(data), api_key)}"
        )
    return _exact(total_tokens, model_info)


def _warn_mistral_approximate(model_name: str) -> str:
    caveat = (
        f"mistral-common ships no tokenizer for {model_name}, so it was counted with "
        "the bundled tekken tokenizer that every Mistral model since Nemo uses. This "
        "count is approximate, not exact. The bundled tokenizers stop at November 2024, "
        "so only a release mistral-common has one for, such as mistral-large-2411, "
        "counts exactly."
    )
    _warn_once("mistral-approximate", model_name, caveat)
    return caveat


def _load_mistral_tokenizer(spec: str) -> MistralTokenizer:
    """Build one of the tokenizers mistral-common bundles, named as in MISTRAL_TOKENIZERS.

    The classmethods are the supported offline path: each reads a tokenizer file shipped
    inside the package, so a count needs no network access and no Hugging Face token.
    mistral-common's own name lookup, `MistralTokenizer.from_model`, is not usable
    instead -- it is deprecated and goes away in mistral-common 1.13.0.
    """
    from mistral_common.tokens.tokenizers.mistral import (  # noqa: PLC0415
        MistralTokenizer,
    )

    version, *flags = spec.split("+")
    options = {f"is_{flag}": True for flag in flags}
    return cast("MistralTokenizer", getattr(MistralTokenizer, version)(**options))


def _count_mistral(text: str, model_info: ModelInfo) -> TokenCount:
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
    except Exception as e:
        raise ValueError(f"Failed to import mistral-common: {e}") from e

    known = MISTRAL_TOKENIZERS.get(model_info.name.lower())
    spec = known or MISTRAL_FALLBACK_TOKENIZER
    caveat = None if known else _warn_mistral_approximate(model_info.name)

    tokenizer = _cached_tokenizer(
        f"mistral:{spec}", lambda: _load_mistral_tokenizer(spec)
    )
    request = ChatCompletionRequest(messages=[UserMessage(content=text)])
    tokens = tokenizer.encode_chat_completion(request).tokens
    if caveat is not None:
        return _approximate(len(tokens), model_info, caveat)
    return _exact(len(tokens), model_info)


def _count_transformers(text: str, model_info: ModelInfo) -> TokenCount:
    if not HAS_TRANSFORMERS:
        raise ValueError(
            f"{model_info.provider.capitalize()} models require the 'transformers' package. "
            "Install with: uv tool install 'toko[transformers]' or uv add 'toko[transformers]'"
        )

    _configure_transformers_logging()

    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = _cached_tokenizer(
            f"transformers:{model_info.name}",
            lambda: AutoTokenizer.from_pretrained(
                model_info.name, trust_remote_code=True
            ),
        )
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

    return _exact(len(tokens), model_info)


_PROVIDER_HANDLERS: dict[str, Callable[[str, ModelInfo], TokenCount]] = {
    "openai": _count_openai,
    "xai": _count_xai,
    "anthropic": _count_anthropic,
    "google": _count_google,
    "mistral": _count_mistral,
}

for provider in ("llama", "deepseek", "qwen"):
    _PROVIDER_HANDLERS[provider] = _count_transformers

_PROVIDER_HANDLERS["huggingface"] = _count_transformers


def count_tokens(text: str, model: str, *, use_cache: bool = True) -> TokenCount:
    """Count tokens in text for a given model.

    Args:
        text: Text to count tokens for
        model: Model name
        use_cache: Whether to use caching (default True)

    Returns:
        The count, the model and provider it resolved to, and whether it is exact

    Raises:
        ValueError: If model is not supported or API key is missing
    """
    # Resolve before the cache lookup: the retirement warning describes the
    # number, not the work of producing it, so a cached count needs it too.
    model_info = get_model(model)
    _warn_if_retired(model_info)

    if use_cache:
        cached = get_cached_count(text, model)
        if cached is not None:
            # Only exact counts are ever stored, so a hit needs no caveat.
            return _exact(cached, model_info)

    result = _count_with_provider(text, model_info)

    # Approximate counts are deliberately not cached. A cache hit returns before any
    # provider runs, so a stored approximation would be replayed on later runs without
    # the stderr warning that says it is not exact.
    if use_cache and not result.approximate:
        cache_count(text, model, result.count)
        if model_info.name != model:
            cache_count(text, model_info.name, result.count)

    return result
