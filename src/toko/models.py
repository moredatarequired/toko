"""Model registry and definitions."""

from collections import defaultdict
from dataclasses import dataclass
from importlib import util

from tiktoken.model import MODEL_TO_ENCODING as TIKTOKEN_MODEL_TO_ENCODING


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    provider: str
    encoding: str | None = None  # For tiktoken models
    api_endpoint: str | None = None  # For API-based counting


def detect_provider(model: str) -> str:
    """Detect provider from model name using pattern matching.

    Args:
        model: Model name

    Returns:
        Provider name (openai, anthropic, google, xai, mistral, llama, deepseek, qwen, unknown)
    """
    model_lower = model.lower()

    # Anthropic Claude models
    if "claude" in model_lower:
        return "anthropic"

    # Google Gemini/Gemma models
    if any(pattern in model_lower for pattern in ["gemini", "gemma"]):
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

    # OpenAI models (GPT, O-series, text-davinci, etc.)
    if any(
        pattern in model_lower
        for pattern in ["gpt-", "o1-", "o3-", "davinci", "curie", "babbage", "ada"]
    ):
        return "openai"

    return "unknown"


# Anthropic models (API-based counting)
_ANTHROPIC_MODELS = [
    # Claude 4.5 family
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    # Claude 4.1/4.0 family
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    # Claude 3.7 family
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest",
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 family (legacy)
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

ANTHROPIC_MODELS = {
    name: ModelInfo(name=name, provider="anthropic") for name in _ANTHROPIC_MODELS
}

# Google models (API-based counting)
# Note: Google API requires "models/" prefix
# Only models that support countTokens are included
_GOOGLE_MODELS = [
    # Gemini 2.5 family
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    # Gemini 2.0 family
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-pro-exp",
    # Aliases
    "gemini-exp-1206",
    "gemini-flash-latest",
    "gemini-pro-latest",
    # Gemma models
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
]

GOOGLE_MODELS = {
    name: ModelInfo(name=f"models/{name}", provider="google") for name in _GOOGLE_MODELS
}

# xAI models (using tiktoken for estimation)
# xAI uses OpenAI-compatible API, use o200k_base as approximation
_XAI_MODELS = [
    "grok-2-1212",
    "grok-2-vision-1212",
    "grok-3",
    "grok-3-mini",
    "grok-4-0709",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "grok-code-fast-1",
]

XAI_MODELS = {
    name: ModelInfo(name=name, provider="xai", encoding="o200k_base")
    for name in _XAI_MODELS
}

# All supported models
# Tokenizer aliases - map common shorthand to actual HuggingFace model paths
# These models share the same tokenizer within their family
TOKENIZER_ALIASES = {
    # Qwen family - all use same tokenizer
    "qwen3": "Qwen/Qwen3-8B",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen2": "Qwen/Qwen2-7B",
    "qwen": "Qwen/Qwen2.5-7B",
    # DeepSeek family
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    # Llama family
    "llama-3.2": "meta-llama/Llama-3.2-1B",
    "llama-3": "meta-llama/Meta-Llama-3-8B",
    "llama": "meta-llama/Llama-3.2-1B",
}

MODELS = {**ANTHROPIC_MODELS, **GOOGLE_MODELS, **XAI_MODELS}


def _has_module(module: str) -> bool:
    return util.find_spec(module) is not None


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
        providers=("llama", "deepseek", "qwen"),
        models=(
            "meta-llama/Llama-3.2-1B",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-7B",
        ),
    ),
)


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

    # Check if it's a tokenizer alias (shorthand name)
    if name.lower() in TOKENIZER_ALIASES:
        canonical_name = TOKENIZER_ALIASES[name.lower()]
        provider = detect_provider(canonical_name)
        return ModelInfo(name=canonical_name, provider=provider)

    # Fall back to dynamic detection
    provider = detect_provider(name)

    if provider == "unknown":
        raise ValueError(
            f"Could not detect provider for model: {name}. "
            "Use --list-models to see known models, or ensure the model name "
            "contains a recognizable provider pattern (claude, gpt, gemini, etc.)"
        )

    # Create ModelInfo based on detected provider
    if provider == "anthropic":
        return ModelInfo(name=name, provider="anthropic")
    if provider == "google":
        # Google API requires "models/" prefix
        model_name = name if name.startswith("models/") else f"models/{name}"
        return ModelInfo(name=model_name, provider="google")
    if provider == "openai":
        # OpenAI-compatible models use tiktoken. encoding determined at runtime.
        return ModelInfo(name=name, provider="openai")
    if provider == "xai":
        return ModelInfo(name=name, provider="xai")
    if provider == "mistral":
        # Mistral uses mistral-common library
        return ModelInfo(name=name, provider="mistral")
    if provider in ("llama", "deepseek", "qwen"):
        # These use HuggingFace transformers tokenizers
        return ModelInfo(name=name, provider=provider)

    raise ValueError(
        f"Provider '{provider}' not supported. "
        f"Supported providers: OpenAI, Anthropic, Google, xAI, Mistral, Llama, DeepSeek, Qwen. "
        f"Model name: {name}"
    )


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
        if provider == "unknown":
            provider = "openai"
        providers[provider].add(model_name)

    return {provider: sorted(models) for provider, models in providers.items()}


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
