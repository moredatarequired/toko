"""Cost estimation using genai-prices."""

import contextlib

from genai_prices import Usage, calc_price

from toko.models import get_model, get_openrouter_id

PROVIDER_ID_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "xai": "xai",
}


def _calculate_price(usage: Usage, *, model_ref: str, provider_id: str) -> float | None:
    with contextlib.suppress(Exception):
        price_data = calc_price(usage, model_ref=model_ref, provider_id=provider_id)
        return float(price_data.total_price)
    return None


def estimate_cost(
    token_count: int, model: str, *, output_tokens: int = 0
) -> float | None:
    """Estimate cost for a given token count and model.

    Args:
        token_count: Number of input tokens
        model: Model name
        output_tokens: Number of output tokens (default 0)

    Returns:
        Total cost in USD, or None if price not available

    Note:
        Prices are estimates and may not be 100% accurate. See genai-prices
        documentation for more information. For models without direct provider
        pricing, falls back to OpenRouter pricing (548 models supported).
    """
    try:
        model_info = get_model(model)
    except ValueError:
        return None
    usage = Usage(input_tokens=token_count, output_tokens=output_tokens)

    # Map provider names to genai-prices provider IDs
    provider_id = PROVIDER_ID_MAP.get(model_info.provider)
    price = None
    if provider_id:
        # The name the user asked for comes first: alias resolution collapses
        # variants onto a canonical name that genai-prices either prices
        # differently or does not know at all (Google's "models/" prefix).
        for model_ref in dict.fromkeys((model, model_info.name)):
            price = _calculate_price(
                usage, model_ref=model_ref, provider_id=provider_id
            )
            if price is not None:
                return price

    openrouter_name = _convert_to_openrouter_name(model_info.name, model_info.provider)
    if openrouter_name is None:
        openrouter_name = get_openrouter_id(model_info.name)

    if openrouter_name:
        price = _calculate_price(
            usage, model_ref=openrouter_name, provider_id="openrouter"
        )
    return price


def _convert_qwen_name(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen3" in lower or "qwen-3" in lower:
        return "qwen/qwen-2.5-72b-instruct"
    if "qwen2.5" in lower or "qwen-2.5" in lower:
        return "qwen/qwen-2.5-72b-instruct"
    if "qwen2" in lower or "qwen-2" in lower:
        return "qwen/qwen-2-72b-instruct"
    return "qwen/qwen-2.5-72b-instruct"


def _convert_deepseek_name(model_name: str) -> str:
    lower = model_name.lower()
    if "r1" in lower or "reasoner" in lower:
        return "deepseek/deepseek-r1"
    return "deepseek/deepseek-chat"


def _convert_llama_name(model_name: str) -> str:
    lower = model_name.lower()
    if "3.3" in model_name:
        return "meta-llama/llama-3.3-70b-instruct"
    if "3.2" in model_name:
        if "90b" in lower:
            return "meta-llama/llama-3.2-90b-vision-instruct"
        if "11b" in lower:
            return "meta-llama/llama-3.2-11b-vision-instruct"
        if "3b" in lower:
            return "meta-llama/llama-3.2-3b-instruct"
        return "meta-llama/llama-3.2-1b-instruct"
    if "3.1" in model_name:
        if "405b" in lower:
            return "meta-llama/llama-3.1-405b-instruct"
        if "70b" in lower:
            return "meta-llama/llama-3.1-70b-instruct"
        return "meta-llama/llama-3.1-8b-instruct"
    if "3" in model_name:
        return "meta-llama/llama-3-8b-instruct"
    return "meta-llama/llama-3.2-1b-instruct"


# Every mistralai/* release genai-prices carries its own entry for, keyed by the bare
# name a user types. The size-word branches below match on substrings, so each one
# swallows the whole tier behind it: without this set mistral-medium-3-5 bills as
# Medium 3 at 27% of its rate and mistral-large-2512 as the four times dearer
# mistral-large. Preferring the exact entry closes that off for every dated release at
# once, including ones added to the data later, rather than one branch at a time.
# test_mistral_exact_ids_cover_the_price_data holds this in step with the shipped data.
_MISTRAL_EXACT_IDS = frozenset(
    {
        "codestral-2501",
        "codestral-2508",
        "codestral-mamba",
        "devstral-2512",
        "ministral-3b",
        "ministral-3b-2512",
        "ministral-8b",
        "ministral-8b-2512",
        "ministral-14b-2512",
        "mistral-7b-instruct",
        "mistral-7b-instruct-v0.1",
        "mistral-7b-instruct-v0.2",
        "mistral-large",
        "mistral-large-2512",
        "mistral-medium-3",
        "mistral-medium-3-5",
        "mistral-medium-3.1",
        "mistral-nemo",
        "mistral-saba",
        "mistral-small",
        "mistral-small-24b-instruct-2501",
        "mistral-small-2603",
        "mistral-small-3.1-24b-instruct",
        "mistral-small-3.2-24b-instruct",
        "mistral-tiny",
        "mixtral-8x22b-instruct",
        "mixtral-8x7b-instruct",
        "pixtral-12b",
        "pixtral-large-2411",
        "voxtral-small-24b-2507",
    }
)

# mistralai/* ids deliberately kept out of the set above. The bare name mistral-medium
# means Medium 3 on Mistral's API and in the native data, which prices any
# mistral-medium* name as Medium 3; OpenRouter's unprefixed entry is instead the retired
# December 2023 release at nearly seven times that, so only mistral-medium-2312 may
# reach it. The ":free" ids are OpenRouter routing suffixes rather than model names, and
# they carry no rate at all.
_MISTRAL_EXACT_ID_EXCLUSIONS = frozenset(
    {
        "mistral-medium",
        "mistral-7b-instruct:free",
        "mistral-nemo:free",
        "mistral-small-24b-instruct-2501:free",
        "mistral-small-3.1-24b-instruct:free",
    }
)


# The data spells release numbers both ways -- mistral-medium-3-5 with a dash sits next
# to mistral-medium-3.1 with a dot -- so a name typed in the other convention misses the
# set and falls onto a size-word rate: mistral-medium-3.5 billed as Medium 3, 3.75x under
# its own entry. Both spellings are looked up, the typed one first. Rewriting every query
# to dashes instead would turn a hit into a miss for the dotted ids that have no dashed
# twin, and a Mistral miss lands on a plausible-looking rate rather than on nothing.
_MISTRAL_EXACT_IDS_BY_DASHED = {
    model_id.replace(".", "-"): model_id for model_id in _MISTRAL_EXACT_IDS
}


def _mistral_exact_id(bare_name: str) -> str | None:
    if bare_name in _MISTRAL_EXACT_IDS:
        return bare_name
    return _MISTRAL_EXACT_IDS_BY_DASHED.get(bare_name.replace(".", "-"))


# Families that carry the same size words as the mistral-* tiers while sharing none of
# their rates, so the branches below read those words as belonging to another model:
# devstral-medium-2507 would bill as mistral-medium-3 and voxtral-mini as mistral-small.
# They skip the ladder and are looked up under the bare name, which is also the spelling
# their entries have -- the OpenRouter data holds devstral-small and the magistral pair
# unprefixed, beside the mistralai/* pair in the set above. A release with no entry
# either way then matches nothing and is reported without a cost, which is the honest
# answer where a neighbouring tier's rate is not. The family word is looked for in that
# bare segment alone, since the branch returns it: matching the whole name would let a
# family word in the org segment turn any last segment into an OpenRouter lookup, so
# devstral/mistral-medium would reach the retired 2312 entry the exclusions above exist
# to keep out.
_MISTRAL_OFF_LADDER_FAMILIES = ("devstral", "magistral", "voxtral")


def _convert_mistral_name(model_name: str) -> str:
    lower = model_name.lower()
    bare = lower.rsplit("/", 1)[-1]
    exact = _mistral_exact_id(bare)
    if exact:
        return f"mistralai/{exact}"
    if any(family in bare for family in _MISTRAL_OFF_LADDER_FAMILIES):
        return bare
    # Left to the size-word branches below, each of these families lands on an unrelated
    # release: open-mixtral-8x7b matches "7b" and bills as mistral-7b-instruct, and any
    # name matching no size word at all falls through to the mistral-small default.
    if "pixtral" in lower:
        return (
            "mistralai/pixtral-large-2411"
            if "large" in lower
            else "mistralai/pixtral-12b"
        )
    # "embed" excluded: codestral-embed-2505 is an embedding model and matching here
    # billed it as the codestral chat release, $0.30/Mtok against its real $0.15. The
    # OpenRouter namespace carries no Mistral embedding entry at all, so it lands on the
    # same mistral-small fallback mistral-embed already uses rather than on a rate that
    # belongs to a different kind of model.
    if "codestral" in lower and "embed" not in lower:
        if "mamba" in lower:
            return "mistralai/codestral-mamba"
        if "2501" in lower:
            return "mistralai/codestral-2501"
        # A rolling codestral name still has to resolve to some release, and the newest
        # one carried beats the mistral-small fallback. codestral-2405 and codestral-22b
        # fall through to that fallback on purpose: neither release is in the data, and
        # 2501/2508 are different models rather than later names for them.
        if "2405" not in lower and "22b" not in lower:
            return "mistralai/codestral-2508"
    if "ministral" in lower and "8b" in lower:
        return "mistralai/ministral-8b"
    if "ministral" in lower and "3b" in lower:
        return "mistralai/ministral-3b"
    if "mixtral" in lower:
        return (
            "mistralai/mixtral-8x22b-instruct"
            if "8x22b" in lower
            else "mistralai/mixtral-8x7b-instruct"
        )
    if "large" in lower:
        return "mistralai/mistral-large"
    if "medium" in lower:
        # The unversioned mistralai/mistral-medium is the retired 2312 release at nearly
        # seven times Medium 3's rate, so only 2312 itself may have it. mistral-medium-3
        # is where the native data sends every other mistral-medium* name, its Medium 3
        # entry matching the prefix, and that is also the only rate native offers for
        # mistral-medium-latest: it has no alias entry and no 3.5 entry to contradict it.
        return (
            "mistralai/mistral-medium"
            if "2312" in lower
            else "mistralai/mistral-medium-3"
        )
    if "small" in lower:
        # Same shape one tier down, but only the rolling name moved on: the dated small
        # releases really are the unversioned entry, while native carries a dedicated
        # mistral-small-latest entry named "Mistral Small 3.2", so the alias means that
        # release. Native prices its alias at $0.10/$0.30 rather than the $0.075/$0.20
        # of its own mistral-small-3.2-24b-instruct entry; toko stays in the OpenRouter
        # namespace, where 3.2 has one rate and the alias and the explicit name agree.
        return (
            "mistralai/mistral-small-3.2-24b-instruct"
            if "latest" in lower
            else "mistralai/mistral-small"
        )
    if "tiny" in lower:
        return "mistralai/mistral-tiny"
    if "nemo" in lower:
        return "mistralai/mistral-nemo"
    if "7b" in lower:
        return "mistralai/mistral-7b-instruct"
    return "mistralai/mistral-small"


_OPENROUTER_CONVERTERS = {
    "qwen": _convert_qwen_name,
    "deepseek": _convert_deepseek_name,
    "llama": _convert_llama_name,
    "mistral": _convert_mistral_name,
}


def _convert_to_openrouter_name(model_name: str, provider: str) -> str | None:
    converter = _OPENROUTER_CONVERTERS.get(provider)
    if converter:
        return converter(model_name)
    return None


def format_cost(cost: float | None) -> str:
    """Format cost for display.

    Args:
        cost: Cost in USD or None

    Returns:
        Formatted cost string
    """
    if cost is None:
        return "N/A"

    # Format with appropriate precision
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_cost_value(cost: float | None) -> str:
    """Render a cost for a machine-readable column: a bare number, or empty for none.

    Not `format_cost`: that one is for people, and its currency symbol and fixed
    decimals turn a cell into something no reader can parse as a number -- a
    fraction of a cent rounds to $0.000000, which reads as free. `g` keeps six
    significant figures wherever the value sits, so the number survives, and
    `float()` accepts every string this produces.
    """
    if cost is None:
        return ""
    return f"{cost:.6g}"
