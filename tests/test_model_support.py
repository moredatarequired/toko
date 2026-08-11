"""Integration tests to ensure listed models are actually usable."""

import importlib.util
import os

import pytest

from tests.hf_hub import skip_if_rate_limited
from toko.counter import count_tokens
from toko.models import get_model, list_models

OPTIONAL_PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
}

OPTIONAL_PROVIDER_MODULES = {
    "mistral": "mistral_common",
    "llama": "transformers",
    "deepseek": "transformers",
    "qwen": "transformers",
    "huggingface": "transformers",
}


def _has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _provider_is_available(provider: str) -> bool:
    env_var = OPTIONAL_PROVIDER_ENV.get(provider)
    if env_var and not os.environ.get(env_var):
        return provider == "xai" and _has_module("transformers")

    module_name = OPTIONAL_PROVIDER_MODULES.get(provider)
    if module_name and not _has_module(module_name):
        return False

    return not (provider == "xai" and not os.environ.get("XAI_API_KEY")) or _has_module(
        "transformers"
    )


@pytest.mark.slow
def test_every_listed_model_counts_tokens():
    models_by_provider = list_models()
    failures: list[str] = []
    text = "integration smoke test"

    with skip_if_rate_limited():
        for provider, models in models_by_provider.items():
            for model in models:
                if not _provider_is_available(provider):
                    continue

                model_info = get_model(model)
                try:
                    count_tokens(text, model_info.name, use_cache=False)
                except Exception as exc:
                    failures.append(f"{model_info.name} ({provider}): {exc}")

    if failures:
        formatted = "\n".join(failures)
        pytest.fail(f"The following models failed to count tokens:\n{formatted}")
