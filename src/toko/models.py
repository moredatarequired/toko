"""Model registry and definitions."""

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    provider: str
    encoding: str | None = None  # For tiktoken models
    api_endpoint: str | None = None  # For API-based counting


# OpenAI models using tiktoken
OPENAI_MODELS = {
    "gpt-4o": ModelInfo(name="gpt-4o", provider="openai", encoding="o200k_base"),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini", provider="openai", encoding="o200k_base"
    ),
    "gpt-4-turbo": ModelInfo(
        name="gpt-4-turbo", provider="openai", encoding="cl100k_base"
    ),
    "gpt-4": ModelInfo(name="gpt-4", provider="openai", encoding="cl100k_base"),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo", provider="openai", encoding="cl100k_base"
    ),
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
