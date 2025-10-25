"""Model registry and definitions."""

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    provider: str
    encoding: str | None = None  # For tiktoken models
    api_endpoint: str | None = None  # For API-based counting


# OpenAI models using tiktoken - o200k_base encoding
_O200K_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-5",
    "gpt-5-turbo",
    "gpt-5-mini",
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-2024-12-17",
    "o3",
    "o3-mini",
]

# OpenAI models using tiktoken - cl100k_base encoding
_CL100K_MODELS = [
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k",
]

# Legacy OpenAI models - p50k_base encoding
_P50K_MODELS = [
    "text-davinci-003",
    "text-davinci-002",
    "code-davinci-002",
]

# Legacy OpenAI models - r50k_base encoding
_R50K_MODELS = [
    "davinci",
    "curie",
    "babbage",
    "ada",
]

OPENAI_MODELS = {
    **{
        name: ModelInfo(name=name, provider="openai", encoding="o200k_base")
        for name in _O200K_MODELS
    },
    **{
        name: ModelInfo(name=name, provider="openai", encoding="cl100k_base")
        for name in _CL100K_MODELS
    },
    **{
        name: ModelInfo(name=name, provider="openai", encoding="p50k_base")
        for name in _P50K_MODELS
    },
    **{
        name: ModelInfo(name=name, provider="openai", encoding="r50k_base")
        for name in _R50K_MODELS
    },
}

# Anthropic models (API-based counting)
_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
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
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-pro-exp",
    "gemini-exp-1206",
    "gemini-flash-latest",
    "gemini-pro-latest",
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
MODELS = {
    **OPENAI_MODELS,
    **ANTHROPIC_MODELS,
    **GOOGLE_MODELS,
    **XAI_MODELS,
}


def get_model(name: str) -> ModelInfo:
    """Get model info by name.

    Args:
        name: Model name

    Returns:
        ModelInfo for the model

    Raises:
        ValueError: If model is not supported
    """
    if name not in MODELS:
        raise ValueError(
            f"Unknown model: {name}. Use --list-models to see supported models."
        )
    return MODELS[name]


def list_models() -> dict[str, list[str]]:
    """List all supported models grouped by provider.

    Returns:
        Dictionary mapping provider name to list of model names
    """
    providers: dict[str, list[str]] = {}
    for model in MODELS.values():
        if model.provider not in providers:
            providers[model.provider] = []
        providers[model.provider].append(model.name)
    return providers
