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

# All supported models
MODELS = {
    **OPENAI_MODELS,
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
