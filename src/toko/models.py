"""Model registry and definitions."""

import json
import re
import sys
import typing
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources, util

from tiktoken.model import MODEL_TO_ENCODING as TIKTOKEN_MODEL_TO_ENCODING


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    provider: str
    encoding: str | None = None  # For tiktoken models
    api_endpoint: str | None = None  # For API-based counting
    # ISO date the provider retired the model, or "unknown" when the provider
    # dropped it without publishing a date. Retired models stay in the registry
    # so toko can explain the failure instead of surfacing a raw 404.
    retired: str | None = None
    # Set when the provider still answers for a retired name but serves a
    # different model, which would otherwise mislabel the count.
    redirects_to: str | None = None
    # Tokenizer generation. Counts are only comparable within one generation, so
    # alias resolution must never map a name across two of these.
    tokenizer: str | None = None


RETIREMENT_DATE_UNKNOWN = "unknown"


_OPENAI_NAME_PATTERN = re.compile(r"(gpt-|o\d)")


def detect_provider(model: str) -> str | None:
    """Detect provider from model name using pattern matching."""
    model_lower = model.lower()
    model_lower_base = model_lower.split("/")[-1]

    for tiktoken_model in TIKTOKEN_MODEL_TO_ENCODING:
        # If tiktoken prefix is in the model name, then the rest should be, e.g.
        # tiktoken includes gpt-5 which covers gpt-5, gpt-5-mini, gpt-5-nano, etc.
        if model_lower_base.startswith(tiktoken_model.lower()):
            return "openai"

    # Anthropic Claude models
    if "claude" in model_lower:
        return "anthropic"

    # Google Gemini/Gemma models
    if any(
        pattern in model_lower for pattern in ["gemini", "gemma"]
    ) and not model_lower.startswith("google/"):
        return "google"

    # xAI Grok models
    if "grok" in model_lower:
        return "xai"

    # Mistral models
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"

    # Llama models (including fine-tunes)
    if "llama" in model_lower:
        return "llama"

    # DeepSeek models
    if "deepseek" in model_lower:
        return "deepseek"

    # Qwen models
    if "qwen" in model_lower:
        return "qwen"

    if model_lower.startswith("gpt-oss"):
        return "huggingface"

    if "/" in model_lower and not model_lower.startswith("models/"):
        return "huggingface"

    # Newer OpenAI names tiktoken has never heard of (gpt-6, gpt-5.6, o5). Checked
    # last so gpt-oss and org-prefixed names keep their more specific providers.
    if _OPENAI_NAME_PATTERN.match(model_lower):
        return "openai"

    return None


# Anthropic models (API-based counting)
#
# Anthropic replaced its tokenizer at Claude Opus 4.7: the same text produces
# roughly 30% more tokens on 4.7-generation models than on everything before
# them. Counting and cost therefore differ per generation, and the two must
# never be conflated -- see _build_anthropic_alias_map.
CLAUDE_TOKENIZER_OPUS_4_7 = "claude-opus-4-7"
CLAUDE_TOKENIZER_LEGACY = "claude-legacy"

# (name, tokenizer, retired date)
_ANTHROPIC_MODEL_SPECS: tuple[tuple[str, str, str | None], ...] = (
    ("claude-fable-5", CLAUDE_TOKENIZER_OPUS_4_7, None),
    ("claude-opus-5", CLAUDE_TOKENIZER_OPUS_4_7, None),
    ("claude-opus-4-8", CLAUDE_TOKENIZER_OPUS_4_7, None),
    ("claude-opus-4-7", CLAUDE_TOKENIZER_OPUS_4_7, None),
    ("claude-sonnet-5", CLAUDE_TOKENIZER_OPUS_4_7, None),
    ("claude-opus-4-6", CLAUDE_TOKENIZER_LEGACY, None),
    ("claude-sonnet-4-6", CLAUDE_TOKENIZER_LEGACY, None),
    ("claude-opus-4-5-20251101", CLAUDE_TOKENIZER_LEGACY, None),
    ("claude-sonnet-4-5-20250929", CLAUDE_TOKENIZER_LEGACY, None),
    ("claude-haiku-4-5-20251001", CLAUDE_TOKENIZER_LEGACY, None),
    ("claude-opus-4-1-20250805", CLAUDE_TOKENIZER_LEGACY, "2026-08-05"),
    ("claude-opus-4-20250514", CLAUDE_TOKENIZER_LEGACY, "2026-06-15"),
    ("claude-sonnet-4-20250514", CLAUDE_TOKENIZER_LEGACY, "2026-06-15"),
    ("claude-3-haiku-20240307", CLAUDE_TOKENIZER_LEGACY, "2026-04-20"),
    ("claude-3-7-sonnet-20250219", CLAUDE_TOKENIZER_LEGACY, "2026-02-19"),
    ("claude-3-5-haiku-20241022", CLAUDE_TOKENIZER_LEGACY, "2026-02-19"),
    ("claude-3-opus-20240229", CLAUDE_TOKENIZER_LEGACY, "2026-01-05"),
    ("claude-3-5-sonnet-20241022", CLAUDE_TOKENIZER_LEGACY, "2025-10-28"),
    ("claude-3-5-sonnet-20240620", CLAUDE_TOKENIZER_LEGACY, "2025-10-28"),
)

ANTHROPIC_MODELS = {
    name: ModelInfo(
        name=name, provider="anthropic", tokenizer=tokenizer, retired=retired
    )
    for name, tokenizer, retired in _ANTHROPIC_MODEL_SPECS
}


def _strip_anthropic_version(name: str) -> str:
    if len(name) > 9 and name[-9] == "-" and name[-8:].isdigit():
        return name[:-9]
    return name


def _build_anthropic_alias_map() -> dict[str, str]:
    candidates: dict[str, list[str]] = defaultdict(list)
    for canonical in ANTHROPIC_MODELS:
        alias = _strip_anthropic_version(canonical)
        if alias != canonical:
            candidates[alias].append(canonical)

    aliases: dict[str, str] = {}
    for alias, names in candidates.items():
        if len({ANTHROPIC_MODELS[name].tokenizer for name in names}) > 1:
            # The alias spans the Opus 4.7 tokenizer change, so resolving it
            # either way would report another generation's token count. Leave it
            # unresolvable and make the user name the model they mean.
            continue
        aliases[alias] = max(names)
    return aliases


_ANTHROPIC_ALIAS_MAP = _build_anthropic_alias_map()


def _resolve_anthropic_model(name: str) -> ModelInfo | None:
    if name in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[name]

    normalized = name.removesuffix("-latest")
    if normalized in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[normalized]

    canonical = _ANTHROPIC_ALIAS_MAP.get(_strip_anthropic_version(normalized))
    if canonical:
        return ANTHROPIC_MODELS[canonical]
    return None


# Google models (API-based counting)
# Note: Google API requires "models/" prefix
# Only models with documented CountTokens support are included
# (name, retired date)
_GOOGLE_MODEL_SPECS: tuple[tuple[str, str | None], ...] = (
    ("gemini-3.1-pro-preview", None),
    ("gemini-3.6-flash", None),
    ("gemini-3.5-flash", None),
    ("gemini-3.5-flash-lite", None),
    ("gemini-3.1-flash-lite", None),
    ("gemini-2.5-pro", None),
    ("gemini-2.5-flash", None),
    ("gemini-2.5-flash-lite", None),
    ("gemini-2.5-flash-image", None),
    ("gemini-3.1-flash-lite-preview", "2026-05-25"),
    ("gemini-3-pro-preview", "2026-03-09"),
    ("gemini-2.0-flash-001", "2026-06-01"),
    ("gemini-2.0-flash-lite-001", "2026-06-01"),
    ("gemini-2.0-flash-preview-image-generation", "2025-11-14"),
)

_GOOGLE_ALIAS_MAP: dict[str, str] = {
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-exp": "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp-02-05": "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp-image-generation": "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-lite-preview": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview-02-05": "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview-image-generation": "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-pro-exp": "gemini-2.5-pro",
    "gemini-2.0-pro-exp-02-05": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
    "gemini-2.5-pro-preview-tts": "gemini-2.5-pro",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025": "gemini-2.5-flash",
    "gemini-2.5-flash-preview-tts": "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image-preview": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview-09-2025": "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite-native-audio-preview-09-2025": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-native-audio-preview-09-2025": "gemini-2.5-flash",
    "gemini-2.5-flash-native-audio-latest": "gemini-2.5-flash",
    "gemini-exp-1206": "gemini-2.5-pro",
    "gemini-flash-latest": "gemini-3.6-flash",
    "gemini-flash-lite-latest": "gemini-3.5-flash-lite",
    "gemini-pro-latest": "gemini-3.1-pro-preview",
}

GOOGLE_MODELS = {
    name: ModelInfo(name=f"models/{name}", provider="google", retired=retired)
    for name, retired in _GOOGLE_MODEL_SPECS
}


def _normalize_google_model_name(name: str) -> str:
    if name.startswith("models/"):
        name = name.split("/", 1)[1]
    lowered = name.lower()
    if lowered in GOOGLE_MODELS:
        return lowered
    alias = _GOOGLE_ALIAS_MAP.get(lowered)
    if alias:
        return alias
    for prefix, canonical in _GOOGLE_ALIAS_MAP.items():
        if lowered.startswith(prefix):
            return canonical
    return lowered


def _resolve_google_model(name: str) -> ModelInfo | None:
    normalized = _normalize_google_model_name(name)
    return GOOGLE_MODELS.get(normalized)


# xAI models (using tiktoken for estimation)
# xAI uses OpenAI-compatible API, use o200k_base as approximation
#
# xAI retired eight slugs on 2026-05-15 but kept them resolving: the API answers
# with a different model instead of 404ing, so a count taken under a retired name
# silently belongs to whatever now serves it. redirects_to records that.
#
# (name, retired date, model served instead)
_XAI_MODEL_SPECS: tuple[tuple[str, str | None, str | None], ...] = (
    ("grok-4.5", None, None),
    ("grok-4.3", None, None),
    ("grok-build-0.1", None, None),
    ("grok-4", "2026-05-15", "grok-4.3"),
    ("grok-4-0709", "2026-05-15", "grok-4.3"),
    ("grok-4-fast-reasoning", "2026-05-15", "grok-4.3"),
    ("grok-4-fast-non-reasoning", "2026-05-15", "grok-4.3"),
    ("grok-4-1-fast-reasoning", "2026-05-15", "grok-4.3"),
    ("grok-4-1-fast-non-reasoning", "2026-05-15", "grok-4.3"),
    ("grok-3", "2026-05-15", "grok-4.3"),
    ("grok-code-fast-1", "2026-05-15", "grok-build-0.1"),
    # Dropped from xAI's model list without a published retirement date.
    ("grok-3-mini", RETIREMENT_DATE_UNKNOWN, None),
    ("grok-2-1212", RETIREMENT_DATE_UNKNOWN, None),
    ("grok-2-vision-1212", RETIREMENT_DATE_UNKNOWN, None),
)

_XAI_ALIAS_MAP = {"grok-2-image-1212": "grok-2-1212"}

XAI_MODELS = {
    name: ModelInfo(
        name=name,
        provider="xai",
        encoding="o200k_base",
        retired=retired,
        redirects_to=redirects_to,
    )
    for name, retired, redirects_to in _XAI_MODEL_SPECS
}


def _resolve_xai_model(name: str) -> ModelInfo | None:
    lowered = name.lower()
    if lowered in XAI_MODELS:
        return XAI_MODELS[lowered]
    alias = _XAI_ALIAS_MAP.get(lowered)
    if alias:
        return XAI_MODELS.get(alias)
    return None


# All supported models
# Tokenizer aliases - map common shorthand to actual HuggingFace model paths
# These models share the same tokenizer within their family
TOKENIZER_ALIASES = {
    # Qwen family - all use same tokenizer
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    # DeepSeek family
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    # Llama family
    "llama-3.1": "NousResearch/Hermes-3-Llama-3.1-8B",
    "llama": "NousResearch/Hermes-3-Llama-3.1-8B",
    # Misc popular open models
    "glm": "THUDM/glm-4-9b-chat",
    "phi": "microsoft/Phi-3.5-mini-instruct",
}

TRANSFORMERS_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "THUDM/glm-4-9b-chat",
    "NousResearch/Hermes-3-Llama-3.1-8B",
)

# tiktoken cannot map dotted OpenAI names to a tokenizer at all, and its prefix
# table only grows on release. Naming an encoding here marks it as verified, so
# counting stays exact and warning-free; anything absent is estimated with
# o200k_base and says so on stderr.
OPENAI_MODEL_ENCODINGS = {
    "gpt-5.1": "o200k_base",
    "gpt-5.1-pro": "o200k_base",
    "gpt-5.2": "o200k_base",
    "gpt-5.2-pro": "o200k_base",
}

# Listed by --list-models on top of tiktoken's own table. gpt-5.1-pro is omitted
# because genai-prices has no entry for it.
POPULAR_OPENAI_MODELS = (
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-pro",
)

MODELS = {**ANTHROPIC_MODELS, **GOOGLE_MODELS, **XAI_MODELS}


def _has_module(module: str) -> bool:
    return util.find_spec(module) is not None


@lru_cache
def _load_openrouter_entries() -> tuple[dict[str, object], ...]:
    try:
        resource = resources.files("toko.data").joinpath("openrouter_models.json")
    except FileNotFoundError:
        return ()
    try:
        payload = json.loads(resource.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return ()
    return tuple(entry for entry in payload if isinstance(entry, dict))


@lru_cache
def _openrouter_id_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in _load_openrouter_entries():
        hugging_face_id = entry.get("hugging_face_id")
        openrouter_id = entry.get("openrouter_id")
        if (
            isinstance(hugging_face_id, str)
            and hugging_face_id
            and isinstance(openrouter_id, str)
            and openrouter_id
        ):
            key = hugging_face_id.lower()
            existing = mapping.get(key)
            if existing and ":" not in existing and ":" in openrouter_id:
                continue
            mapping[key] = openrouter_id
    return mapping


def get_openrouter_id(model_name: str) -> str | None:
    return _openrouter_id_map().get(model_name.lower())


@dataclass(frozen=True)
class OptionalGroupDef:
    extra: str
    module: str
    providers: tuple[str, ...]
    models: tuple[str, ...]


OPTIONAL_GROUPS: tuple[OptionalGroupDef, ...] = (
    OptionalGroupDef(
        extra="mistral",
        module="mistral_common",
        providers=("mistral",),
        models=(
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
        ),
    ),
    OptionalGroupDef(
        extra="transformers",
        module="transformers",
        providers=("llama", "deepseek", "qwen", "huggingface"),
        models=TRANSFORMERS_MODELS[:3],
    ),
)


def _make_basic_builder(provider: str) -> typing.Callable[[str], ModelInfo]:
    def builder(name: str) -> ModelInfo:
        return ModelInfo(name=name, provider=provider)

    return builder


def _build_openai_model(name: str) -> ModelInfo:
    return ModelInfo(
        name=name, provider="openai", encoding=OPENAI_MODEL_ENCODINGS.get(name.lower())
    )


def _build_google_model(name: str) -> ModelInfo:
    if name.startswith("models/"):
        model_name = name
    elif "/" in name:
        model_name = f"models/{name.split('/', 1)[1]}"
    else:
        model_name = f"models/{name}"
    return ModelInfo(name=model_name, provider="google")


_PROVIDER_BUILDERS: dict[str, typing.Callable[[str], ModelInfo]] = {
    "anthropic": _make_basic_builder("anthropic"),
    "google": _build_google_model,
    "openai": _build_openai_model,
    "xai": _make_basic_builder("xai"),
    "mistral": _make_basic_builder("mistral"),
    "llama": _make_basic_builder("llama"),
    "deepseek": _make_basic_builder("deepseek"),
    "qwen": _make_basic_builder("qwen"),
    "huggingface": _make_basic_builder("huggingface"),
}


def get_model(name: str) -> ModelInfo:
    """Get model info by name.

    Tries to find model in registry first, then checks tokenizer aliases,
    then falls back to dynamic detection.

    Args:
        name: Model name (can be full path or shorthand alias)

    Returns:
        ModelInfo for the model

    Raises:
        ValueError: If model provider cannot be detected
    """
    # First try the registry
    if name in MODELS:
        return MODELS[name]

    resolved_anthropic = _resolve_anthropic_model(name)
    if resolved_anthropic is not None:
        return resolved_anthropic

    resolved_google = _resolve_google_model(name)
    if resolved_google is not None:
        return resolved_google

    resolved_xai = _resolve_xai_model(name)
    if resolved_xai is not None:
        return resolved_xai

    # Check if it's a tokenizer alias (shorthand name)
    if name.lower() in TOKENIZER_ALIASES:
        canonical_name = TOKENIZER_ALIASES[name.lower()]
        provider = detect_provider(canonical_name) or "openai"
        return ModelInfo(name=canonical_name, provider=provider)

    # Fall back to dynamic detection
    provider = detect_provider(name)

    if provider is None:
        raise ValueError(
            f"Could not detect provider for model: {name}. "
            "Use --list-models to see known models, or ensure the model name "
            "contains a recognizable provider pattern (claude, gpt, gemini, etc.)"
        )

    builder = _PROVIDER_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            "Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Llama, DeepSeek, Qwen. "
            f"Model name: {name}"
        )

    return builder(name)


def retirement_notice(model_info: ModelInfo) -> str | None:
    """Explain that a model is retired, and what the provider serves instead."""
    if model_info.retired is None:
        return None
    when = (
        "on an unpublished date"
        if model_info.retired == RETIREMENT_DATE_UNKNOWN
        else f"on {model_info.retired}"
    )
    notice = f"Warning: {model_info.name} was retired {when}"
    if model_info.redirects_to:
        return (
            f"{notice}; {model_info.provider} still answers for it but serves "
            f"{model_info.redirects_to}, so this count is {model_info.redirects_to}'s, "
            f"not {model_info.name}'s."
        )
    return f"{notice}; the {model_info.provider} API will reject or redirect it."


def warn_if_retired(model_info: ModelInfo) -> None:
    notice = retirement_notice(model_info)
    if notice is not None:
        print(notice, file=sys.stderr)


def list_models(*, include_retired: bool = False) -> dict[str, list[str]]:
    """List all supported models grouped by provider.

    Args:
        include_retired: Also list models the provider has retired. They stay in
            the registry so toko can explain the failure, but they are hidden by
            default because they can no longer be counted.

    Returns:
        Dictionary mapping provider name to list of model names
    """
    providers: dict[str, set[str]] = defaultdict(set)

    for model in MODELS.values():
        if model.retired is not None and not include_retired:
            continue
        providers[model.provider].add(model.name)

    for model_name in TIKTOKEN_MODEL_TO_ENCODING:
        provider = detect_provider(model_name)
        if provider is None:
            provider = "openai"
        providers[provider].add(model_name)

    if _has_module("transformers"):
        for model_name in TRANSFORMERS_MODELS:
            provider = detect_provider(model_name) or "huggingface"
            providers[provider].add(model_name)

    for alias in POPULAR_OPENAI_MODELS:
        providers["openai"].add(alias)

    return {
        provider: sorted(models, key=str.lower)
        for provider, models in providers.items()
    }


def list_optional_model_groups() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for group in OPTIONAL_GROUPS:
        installed = _has_module(group.module)
        groups.append(
            {
                "extra": group.extra,
                "providers": list(group.providers),
                "models": list(group.models),
                "installed": installed,
            }
        )
    return groups
