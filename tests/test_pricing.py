"""Comprehensive pricing tests for all supported models."""

import pytest
from genai_prices.data import providers

from toko import models
from toko.cost import (
    _MISTRAL_EXACT_ID_EXCLUSIONS,
    _MISTRAL_EXACT_IDS,
    _MISTRAL_EXACT_IDS_BY_DASHED,
    _MISTRAL_OFF_LADDER_FAMILIES,
    _convert_mistral_name,
    _mistral_exact_id,
    estimate_cost,
)
from toko.models import (
    ANTHROPIC_MODELS,
    GOOGLE_MODELS,
    UNLISTED_OPENAI_MODELS,
    XAI_MODELS,
    detect_provider,
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


# The devstral, magistral and voxtral ids the OpenRouter data carries a rate for, taken
# from the data so a release added to it is covered without touching this file. The
# ":free" ids are routing suffixes rather than models, as _MISTRAL_EXACT_ID_EXCLUSIONS
# explains, and carry no rate to assert.
_OFF_LADDER_PRICED_IDS = sorted(
    model.id
    for provider in providers
    if provider.id == "openrouter"
    for model in provider.models
    if ":" not in model.id
    and any(family in model.id for family in _MISTRAL_OFF_LADDER_FAMILIES)
)


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
        failures = []
        openai_models = list_models(include_retired=True).get("openai", [])

        current_prefixes = ("gpt-", "o", "text-embedding-3")

        for model_name in openai_models:
            if model_name in UNLISTED_OPENAI_MODELS:
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

    @pytest.mark.parametrize(
        "model_name",
        [
            "mistral-large-2411",
            "mistral-small-2409",
            "mistral-nemo",
            "open-mistral-7b",
            "codestral-latest",
            "codestral-2501",
            "codestral-2508",
            "codestral-mamba-2407",
            "ministral-3b",
            "ministral-8b-2410",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-tiny-2407",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "pixtral-12b-2409",
            "pixtral-large-2411",
            "mistral-medium-3-5",
            "mistral-large-2512",
            "mistral-small-2603",
            "mistral-small-3.1-24b-instruct",
            "ministral-14b-2512",
            "codestral-embed-2505",
            "devstral-2512",
            "devstral-small",
            "magistral-small-2506",
            "magistral-medium-2506",
            "voxtral-small-24b-2507",
        ],
    )
    def test_mistral_family_models_have_pricing(self, model_name):
        """Every OpenRouter id _convert_mistral_name can produce has to be a real one.

        genai-prices matches these exactly, so an id that only looks plausible prices
        nothing at all.
        """
        assert estimate_cost(1_000_000, model_name) is not None

    @pytest.mark.parametrize(
        ("model_name", "other_name"),
        [
            ("ministral-3b", "mistral-small-2409"),
            ("mistral-tiny-2407", "mistral-small-2409"),
            ("open-mixtral-8x7b", "open-mistral-7b"),
            ("open-mixtral-8x22b", "mistral-small-2409"),
            ("codestral-2501", "codestral-2405"),
            ("mistral-medium-latest", "mistral-medium-2312"),
            ("mistral-small-latest", "mistral-small-2409"),
            ("pixtral-12b-2409", "mistral-small-2409"),
            # A dated release must not collapse onto the rolling name of its tier: each
            # of these matched the tier's size word and billed as the wrong release.
            ("mistral-medium-3-5", "mistral-medium-latest"),
            ("mistral-large-2512", "mistral-large-2411"),
            ("mistral-small-2603", "mistral-small-2409"),
            ("mistral-small-3.1-24b-instruct", "mistral-small-2409"),
            ("mistral-small-3.1-24b-instruct", "mistral-small-3.2-24b-instruct"),
            ("mistral-small-24b-instruct-2501", "mistral-small-2409"),
            ("ministral-3b-2512", "ministral-3b"),
            ("ministral-8b-2512", "ministral-8b-2410"),
            ("ministral-14b-2512", "mistral-small-2409"),
            ("mistral-7b-instruct-v0.2", "mistral-7b-instruct"),
            # An embedding model must not be billed at its family's chat rate.
            ("codestral-embed-2505", "codestral-2508"),
            # The families that share the tier words without sharing the tiers: each of
            # these reads as a mistral-small or mistral-medium to the size-word branches.
            ("devstral-2512", "mistral-small-2409"),
            ("devstral-small", "mistral-small-2409"),
            ("magistral-small-2506", "mistral-small-2409"),
            ("magistral-medium-2506", "mistral-medium-latest"),
            ("voxtral-small-24b-2507", "mistral-small-2409"),
        ],
    )
    def test_mistral_family_models_are_not_priced_as_each_other(
        self, model_name, other_name
    ):
        """Distinct models must not collapse onto one id, which pricing alone hides.

        Every name here used to fall through _convert_mistral_name onto the id its
        partner resolves to, so each priced to something and the swap went unnoticed.
        Comparing the two survives a price update in a way a hardcoded rate would not.
        Both directions are priced because some pairs share an input rate and differ
        only on output, ministral-14b-2512 against the mistral-small fallback among them.
        """
        assert estimate_cost(1_000_000, model_name, output_tokens=1_000_000) != (
            estimate_cost(1_000_000, other_name, output_tokens=1_000_000)
        )

    def test_mistral_exact_ids_cover_the_price_data(self):
        """_MISTRAL_EXACT_IDS has to keep up with the shipped price data.

        A mistralai/* release the set misses falls back to a size-word branch and bills
        as the wrong member of its tier, which is the defect the set exists to prevent,
        so a data update adding one has to fail here rather than pass quietly.
        """
        available = {
            model.id.removeprefix("mistralai/")
            for provider in providers
            if provider.id == "openrouter"
            for model in provider.models
            if model.id.startswith("mistralai/")
        }
        # Every bare id in the namespace now detects as Mistral and so reaches
        # _convert_mistral_name, which is what lets the whole set be asserted against.
        # A family the patterns miss lands in the difference below rather than being
        # quietly filtered out of it.
        assert all(detect_provider(name) == "mistral" for name in available)
        assert available - _MISTRAL_EXACT_IDS == _MISTRAL_EXACT_ID_EXCLUSIONS
        assert available >= _MISTRAL_EXACT_IDS

    @pytest.mark.parametrize("model_id", sorted(_MISTRAL_EXACT_IDS))
    def test_every_exact_id_resolves_to_itself(self, model_id):
        """Ids that bill alike need the id asserted; no price comparison can see them.

        Several members of the set share both rates: pixtral-large-2411 with
        mistral-large, and codestral-2501 with codestral-2508, which differ only on the
        cache read rate estimate_cost is never asked about above. Collapsing either onto
        its twin costs nothing and every comparison above still passes. Covering the
        whole set rather than the pairs that happen to coincide today keeps that true
        when a price update makes two more of them agree.
        """
        assert _convert_mistral_name(model_id) == f"mistralai/{model_id}"

    @pytest.mark.parametrize(
        ("model_name", "expected_id"),
        [
            ("pixtral-large", "mistralai/pixtral-large-2411"),
            ("pixtral-large-latest", "mistralai/pixtral-large-2411"),
            ("pixtral", "mistralai/pixtral-12b"),
            ("pixtral-12b-2409", "mistralai/pixtral-12b"),
            ("codestral-mamba-2407", "mistralai/codestral-mamba"),
            # The dated releases are all in the set, so the 2501 arm is only reachable
            # through a vendor spelling of one; the data spells Bedrock's Large this way.
            ("mistral.codestral-2501-v1:0", "mistralai/codestral-2501"),
            ("codestral-latest", "mistralai/codestral-2508"),
            ("codestral", "mistralai/codestral-2508"),
            # The two releases genai-prices does not carry keep the fallback.
            ("codestral-2405", "mistralai/mistral-small"),
            ("codestral-22b", "mistralai/mistral-small"),
            ("ministral-8b-2410", "mistralai/ministral-8b"),
            ("mistral.ministral-3-3b-instruct", "mistralai/ministral-3b"),
            ("open-mixtral-8x22b", "mistralai/mixtral-8x22b-instruct"),
            ("open-mixtral-8x7b", "mistralai/mixtral-8x7b-instruct"),
            ("mistral-large-2411", "mistralai/mistral-large"),
            ("mistral-medium-2312", "mistralai/mistral-medium"),
            ("mistral-medium-latest", "mistralai/mistral-medium-3"),
            ("mistral-small-latest", "mistralai/mistral-small-3.2-24b-instruct"),
            ("mistral-small-2409", "mistralai/mistral-small"),
            ("mistral-tiny-2407", "mistralai/mistral-tiny"),
            ("open-mistral-nemo-2407", "mistralai/mistral-nemo"),
            ("open-mistral-7b", "mistralai/mistral-7b-instruct"),
            # The trailing fallback, which no size word reaches.
            ("mistral-embed", "mistralai/mistral-small"),
            ("codestral-embed-2505", "mistralai/mistral-small"),
        ],
    )
    def test_size_word_branches_resolve_names_the_exact_ids_miss(
        self, model_name, expected_id
    ):
        """One name per size-word branch, none of them answered by the exact-id set.

        The set is consulted first, so an assertion whose subject is a member tests the
        lookup and leaves the branch behind it free to return anything. That is how three
        of these branches lost their cover: pixtral-large-2411, codestral-2501 and
        ministral-3b each guarded one until they joined the set. The membership check
        below fails rather than goes quiet if a later addition steals a subject the same
        way. Ids are asserted rather than prices because several branches return rates
        that coincide, on the output side especially.
        """
        assert _mistral_exact_id(model_name.rsplit("/", 1)[-1]) is None
        assert _convert_mistral_name(model_name) == expected_id

    @pytest.mark.parametrize(
        ("model_name", "expected_id"),
        [
            ("mistral-medium-3-5", "mistralai/mistral-medium-3-5"),
            ("mistral-medium-3.5", "mistralai/mistral-medium-3-5"),
            ("mistral-medium-3.1", "mistralai/mistral-medium-3.1"),
            ("mistral-medium-3-1", "mistralai/mistral-medium-3.1"),
            (
                "mistral-small-3-2-24b-instruct",
                "mistralai/mistral-small-3.2-24b-instruct",
            ),
            (
                "mistralai/mistral-7b-instruct-v0-2",
                "mistralai/mistral-7b-instruct-v0.2",
            ),
        ],
    )
    def test_release_numbers_resolve_in_either_spelling(self, model_name, expected_id):
        """The data uses both conventions, so a release has to answer to both.

        mistral-medium-3-5 is dashed and mistral-medium-3.1 dotted, and typing the other
        spelling used to miss the set entirely: mistral-medium-3.5 billed as Medium 3 at
        27% of its own rate. The dotted-only ids are here so that a one-way rewrite to
        dashes, which would drop them back onto a size-word branch, fails.
        """
        assert _convert_mistral_name(model_name) == expected_id

    def test_dotted_and_dashed_spellings_do_not_collide(self):
        """Two ids differing only in dots would leave one spelling unreachable."""
        assert len(_MISTRAL_EXACT_IDS_BY_DASHED) == len(_MISTRAL_EXACT_IDS)

    def test_the_off_ladder_families_are_all_represented_in_the_price_data(self):
        """A family with no entries left would make the test below pass with no cases."""
        assert {
            family
            for family in _MISTRAL_OFF_LADDER_FAMILIES
            for model_id in _OFF_LADDER_PRICED_IDS
            if family in model_id
        } == set(_MISTRAL_OFF_LADDER_FAMILIES)

    @pytest.mark.parametrize("model_id", _OFF_LADDER_PRICED_IDS)
    def test_every_off_ladder_id_in_the_price_data_is_priced(self, model_id):
        """These three families are the ones the data spells inconsistently.

        mistralai/devstral-2512 carries the prefix every other Mistral id there has and
        magistral-small-2506 does not, while a model_ref matches by exact id, so each id
        has to be produced in its own spelling. Enumerating the data rather than listing
        names means a release it gains has to be spelled right too.
        """
        assert estimate_cost(1_000_000, model_id) is not None

    def test_a_family_word_in_the_org_segment_is_not_a_passthrough(self):
        """The branch returns the last segment, so it has to match on that segment.

        Reading the whole name instead let any devstral/* org hand an unrelated id
        straight to the OpenRouter lookup, mistral-medium among them -- the one entry
        _MISTRAL_EXACT_ID_EXCLUSIONS exists to reserve for mistral-medium-2312.
        """
        foreign = [
            model.id
            for provider in providers
            if provider.id == "openrouter"
            for model in provider.models
            if "/" not in model.id
            and not any(family in model.id for family in _MISTRAL_OFF_LADDER_FAMILIES)
        ]
        assert foreign
        assert not [
            model_id
            for model_id in foreign
            if _convert_mistral_name(f"devstral/{model_id}") == model_id
        ]
        assert (
            _convert_mistral_name("devstral/mistral-medium")
            == "mistralai/mistral-medium-3"
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "devstral-small-2505",
            "devstral-small-2507",
            "devstral-medium-2507",
            "magistral-medium-2507",
            "magistral-medium-2509",
            "voxtral-mini-2507",
            "voxtral-small-2507",
        ],
    )
    def test_off_ladder_releases_the_price_data_lacks_cost_nothing(self, model_name):
        """No price is the honest answer for these; a neighbouring tier's rate is not.

        Each is a real release genai-prices carries no entry for, and each reads as a
        mistral-small or mistral-medium to the size-word branches -- which is what they
        billed as when detection first started routing them to Mistral. The 2505 name is
        here rather than under the priced ids because the data's devstral-small row
        entered in June 2025, before 1.1 existed, and still carries 2505's description,
        but OpenRouter has since reassigned that slug to 1.1 and the row was never
        re-scraped: its $0.06/$0.12 aggregate is wrong for both releases, which Mistral
        priced alike at $0.10/$0.30. estimate_cost is asserted rather
        than the converter's id: what a miss has to produce is no cost, and only the
        whole path shows that a miss stays a miss.
        """
        assert detect_provider(model_name) == "mistral"
        assert estimate_cost(1_000_000, model_name, output_tokens=1_000_000) is None

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
