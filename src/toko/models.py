"""Model registry and definitions."""

import json
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

    return None


# Anthropic models (API-based counting)
_ANTHROPIC_MODELS = [
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
]

ANTHROPIC_MODELS = {
    name: ModelInfo(name=name, provider="anthropic") for name in _ANTHROPIC_MODELS
}


def _strip_anthropic_version(name: str) -> str:
    if len(name) > 9 and name[-9] == "-" and name[-8:].isdigit():
        return name[:-9]
    return name


_ANTHROPIC_ALIAS_MAP: dict[str, str] = {}
for canonical in _ANTHROPIC_MODELS:
    alias = _strip_anthropic_version(canonical)
    existing = _ANTHROPIC_ALIAS_MAP.get(alias)
    if existing is None or canonical > existing:
        _ANTHROPIC_ALIAS_MAP[alias] = canonical


def _resolve_anthropic_model(name: str) -> ModelInfo | None:
    if name in ANTHROPIC_MODELS:
        return ANTHROPIC_MODELS[name]

    normalized = name
    normalized = normalized.removesuffix("-latest")

    base = _strip_anthropic_version(normalized)
    canonical = _ANTHROPIC_ALIAS_MAP.get(base)
    if canonical:
        return ANTHROPIC_MODELS[canonical]
    return None


# Google models (API-based counting)
# Note: Google API requires "models/" prefix
# Only models with documented CountTokens support are included
_GOOGLE_CANONICAL_MODELS = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-preview-image-generation",
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
    "gemini-flash-latest": "gemini-2.5-flash",
    "gemini-flash-lite-latest": "gemini-2.5-flash-lite",
    "gemini-pro-latest": "gemini-2.5-pro",
}

GOOGLE_MODELS = {
    name: ModelInfo(name=f"models/{name}", provider="google")
    for name in _GOOGLE_CANONICAL_MODELS
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
_XAI_MODELS = (
    "grok-4",
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-3",
    "grok-3-mini",
    "grok-2-1212",
    "grok-2-vision-1212",
    "grok-code-fast-1",
)

_XAI_ALIAS_MAP = {"grok-4-0709": "grok-4", "grok-2-image-1212": "grok-2-1212"}

XAI_MODELS = {
    name: ModelInfo(name=name, provider="xai", encoding="o200k_base")
    for name in _XAI_MODELS
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

OPENAI_MODEL_FALLBACKS = {
    "gpt-4.1-mini": "gpt-4.1",
    "gpt-4.1-nano": "gpt-4.1",
    "gpt-5-mini": "gpt-5",
    "gpt-5-nano": "gpt-5",
    "gpt-5.1": "gpt-5",
    "gpt-5.1-pro": "gpt-5",
    "gpt-5.2": "gpt-5",
    "gpt-5.2-pro": "gpt-5",
}

POPULAR_OPENAI_MODELS = tuple(sorted(OPENAI_MODEL_FALLBACKS))

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
    "openai": _make_basic_builder("openai"),
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

    lower_name = name.lower()
    if lower_name in OPENAI_MODEL_FALLBACKS:
        canonical_name = OPENAI_MODEL_FALLBACKS[lower_name]
        return ModelInfo(name=canonical_name, provider="openai")

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


def list_models() -> dict[str, list[str]]:
    """List all supported models grouped by provider.

    Returns:
        Dictionary mapping provider name to list of model names
    """
    providers: dict[str, set[str]] = defaultdict(set)

    for model in MODELS.values():
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
