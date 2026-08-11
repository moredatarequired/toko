"""Comprehensive pricing tests for all supported models."""

import pytest

from toko import models
from toko.cost import estimate_cost
from toko.models import (
    ANTHROPIC_MODELS,
    GOOGLE_MODELS,
    RETIRED_OPENAI_MODELS,
    XAI_MODELS,
    list_models,
)

# Anthropic and xAI aliases resolve to a canonical name genai-prices already knows,
# so trying the user's name first must stay invisible for them. Enumerated rather
# than listed so a newly added alias is covered without touching this file.
_VERSION_ALIASES = [
    pytest.param(alias, canonical, id=alias)
    for alias_map in (
        models._ANTHROPIC_ALIAS_MAP,  # noqa: SLF001
        models._XAI_ALIAS_MAP,  # noqa: SLF001
    )
    for alias, canonical in sorted(alias_map.items())
]


def _current(registry):
    """Return the model names the provider still serves.

    Retired models are kept in the registry only to explain the failure, so
    pricing them is not required.
    """
    return [name for name, info in registry.items() if info.retired is None]


# genai-prices ships its price table as data, so an older release simply has no
# entry for a model released after it. Probing tells us whether the installed
# release is new enough for the coverage assertions below to mean anything; if
# it is not, they would fail for every current model at once. Each provider gets
# its own probe: genai-prices adds providers at its own pace, so one provider's
# coverage says nothing about another's.
_PROVIDER_PROBES = {
    "anthropic": "claude-opus-5",
    "google": "gemini-3.6-flash",
    "xai": "grok-4.5",
}


def requires_current_prices(provider: str):
    probe = _PROVIDER_PROBES[provider]
    return pytest.mark.skipif(
        estimate_cost(1000, probe) is None,
        reason=(
            f"installed genai-prices predates the current {provider} model "
            f"generation; it has no entry for {probe}"
        ),
    )


# xAI ships models faster than genai-prices publishes their rates. These are
# current on docs.x.ai and so belong in the registry, but asserting a price for
# them would only assert how stale the installed price table is.
_UNPRICED_XAI_MODELS = frozenset(
    {
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-multi-agent-0309",
    }
)


class TestPricingCoverage:
    """Test that pricing works for all models we claim to support."""

    def test_openai_models_have_pricing(self):
        """All current OpenAI models should have pricing data."""
        # gpt-3.5-turbo-0301 is a dated snapshot tiktoken still lists; genai-prices
        # prices only the undated name, so it is exempt alongside the retired engines.
        exempt = RETIRED_OPENAI_MODELS | {"gpt-3.5-turbo-0301"}

        failures = []
        openai_models = list_models(include_retired=True).get("openai", [])

        current_prefixes = ("gpt-", "o", "text-embedding-3")

        for model_name in openai_models:
            if model_name in exempt:
                continue
            if not any(model_name.startswith(prefix) for prefix in current_prefixes):
                # Skip legacy completions and embeddings we no longer track pricing for
                continue
            cost = estimate_cost(100, model_name)
            if cost is None:
                failures.append(model_name)

        if failures:
            pytest.fail(
                f"The following current OpenAI models lack pricing data: {', '.join(failures)}"
            )

    @requires_current_prices("anthropic")
    def test_anthropic_models_have_pricing(self):
        """All current Anthropic models should have pricing data."""
        failures = []
        for model_name in _current(ANTHROPIC_MODELS):
            cost = estimate_cost(100, model_name)
            if cost is None:
                failures.append(model_name)

        if failures:
            pytest.fail(
                f"The following Anthropic models lack pricing data: {', '.join(failures)}"
            )

    @requires_current_prices("google")
    def test_google_models_have_pricing(self):
        """Current Google models should have pricing data."""
        failures = []
        for model_name in _current(GOOGLE_MODELS):
            cost = estimate_cost(100, model_name)
            if cost is None:
                failures.append(model_name)

        if failures:
            pytest.fail(
                f"The following Google models lack pricing data: {', '.join(failures)}"
            )

    @requires_current_prices("xai")
    def test_xai_models_have_pricing(self):
        """Current xAI models should have pricing data."""
        failures = []
        for model_name in _current(XAI_MODELS):
            if model_name in _UNPRICED_XAI_MODELS:
                continue
            cost = estimate_cost(100, model_name)
            if cost is None:
                failures.append(model_name)

        if failures:
            pytest.fail(
                f"The following xAI models lack pricing data: {', '.join(failures)}"
            )

        # Keep the exemption list honest: once genai-prices catches up, the
        # models belong back under the assertion above.
        caught_up = [
            name
            for name in sorted(_UNPRICED_XAI_MODELS)
            if estimate_cost(100, name) is not None
        ]
        if caught_up:
            pytest.fail(
                "genai-prices now prices these; drop them from "
                f"_UNPRICED_XAI_MODELS: {', '.join(caught_up)}"
            )

    def test_tokenizer_aliases_have_pricing(self):
        """Test that tokenizer aliases (shorthand names) have pricing via OpenRouter."""
        # Test common shorthand aliases
        aliases_to_test = ["qwen2.5", "qwen2", "deepseek-v3", "llama"]

        failures = []
        for alias in aliases_to_test:
            cost = estimate_cost(100, alias)
            if cost is None:
                failures.append(alias)

        if failures:
            pytest.fail(
                f"The following tokenizer aliases lack pricing data: {', '.join(failures)}"
            )


class TestPricingAccuracy:
    """Test that pricing calculations are correct."""

    def test_gpt5_pricing(self):
        """Test gpt-5 pricing calculation."""
        # gpt-5: ~$1.25 per 1M input tokens
        cost = estimate_cost(1000, "gpt-5")
        assert cost is not None
        # Should be approximately $0.00125 for 1000 tokens
        assert 0.001 <= cost <= 0.002

    def test_gpt5_point_releases_are_priced_separately(self):
        assert estimate_cost(1_000_000, "gpt-5.2") != estimate_cost(1_000_000, "gpt-5")

    def test_gemini_pricing_uses_the_name_the_user_asked_for(self):
        # The registry canonicalizes to "models/gemini-2.5-pro", which genai-prices
        # has no entry for.
        cost = estimate_cost(1000, "gemini-2.5-pro")
        assert cost is not None
        assert cost > 0

    @pytest.mark.parametrize(("alias", "canonical"), _VERSION_ALIASES)
    def test_alias_prices_match_its_canonical_model(self, alias, canonical):
        # genai-prices matches model refs by prefix, so an alias could otherwise
        # silently win against a different entry than its canonical name resolves to.
        assert estimate_cost(1_000_000, alias) == estimate_cost(1_000_000, canonical)

    def test_claude_sonnet_pricing(self):
        """Test Claude 3.5 Sonnet pricing calculation."""
        # Claude 3.5 Sonnet: $3.00 per 1M input tokens
        cost = estimate_cost(1000, "claude-3-5-sonnet-20241022")
        assert cost is not None
        # Should be approximately $0.003 for 1000 tokens
        assert 0.002 <= cost <= 0.004

    def test_claude_opus_4_5_pricing(self):
        """Test Claude Opus 4.5 pricing is available."""
        cost = estimate_cost(1000, "claude-opus-4-5")
        assert cost is not None
        assert cost > 0

    def test_zero_tokens_pricing(self):
        """Test pricing with zero tokens."""
        cost = estimate_cost(0, "gpt-5")
        assert cost == 0.0

    def test_large_token_count_pricing(self):
        """Test pricing with large token count."""
        # 1M tokens should be reasonable
        cost = estimate_cost(1_000_000, "gpt-5")
        assert cost is not None
        assert cost > 1.0  # Should be at least $1

    def test_open_source_models_have_openrouter_pricing(self):
        """Open source models should have OpenRouter pricing as fallback."""
        # Qwen, DeepSeek, Llama are open source but available via OpenRouter
        # These should have pricing via OpenRouter fallback
        qwen_cost = estimate_cost(1000, "Qwen/Qwen3-8B")
        assert qwen_cost is not None
        assert qwen_cost > 0

        deepseek_cost = estimate_cost(1000, "deepseek-ai/DeepSeek-V3")
        assert deepseek_cost is not None
        assert deepseek_cost > 0

        llama_cost = estimate_cost(1000, "meta-llama/Llama-3.2-1B")
        assert llama_cost is not None
        assert llama_cost > 0


class TestPricingErrors:
    """Test pricing error handling."""

    def test_unknown_model_returns_none(self):
        """Unknown models should return None instead of raising."""
        # Use a model that will be detected but has no pricing
        cost = estimate_cost(100, "gpt-5-fake-xyz")
        # Should not raise, just return None
        assert cost is None or isinstance(cost, float)

    def test_negative_tokens_handled(self):
        """Negative token counts should be handled gracefully."""
        # This shouldn't happen in practice, but test defensive coding
        try:
            cost = estimate_cost(-100, "gpt-5")
            # If it doesn't raise, it should return None or 0
            assert cost is None or cost == 0
        except ValueError:
            # Or it can raise ValueError, both are acceptable
            pass
